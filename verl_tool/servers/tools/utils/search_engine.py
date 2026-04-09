import os
import json
import time
import pathlib
import asyncio
import aiofiles
import aiohttp
from typing import Optional, Union, Dict, List, Any, Tuple
from collections import OrderedDict
from urllib.parse import quote

import regex as re
import faulthandler
from .deepsearch_utils import extract_relevant_info_serper, extract_text_from_url, extract_snippet_with_context
from .web_agent_utils import generate_webpage_to_reasonchain, get_prev_reasoning_chain

import logging
import threading

# --- Global Configuration ---
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "brightdata")  # Can be 'serpapi' or 'brightdata'
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
BRIGHTDATA_API_TOKEN = os.getenv("BRIGHTDATA_API_TOKEN", "")
BRIGHTDATA_ZONE = os.getenv("BRIGHTDATA_ZONE", "")

faulthandler.enable()
DEBUG = False
logger = logging.getLogger(__name__)

class AsyncLRUCache:
    """Thread-safe LRU cache for async operations"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                # Check TTL
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    del self._cache[key]
                    del self._timestamps[key]
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            return None
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            
            self._cache[key] = value
            self._timestamps[key] = time.time()


class GoogleSearchEngine:
    """
    Async Google search engine supporting multiple backends and batch queries.
    """

    def __init__(
        self,
        api_key: str,
        provider: str = SEARCH_PROVIDER,
        max_results: int = 10,
        result_length: int = 20000,
        cache_file: Optional[str] = None,
        process_snippets: bool = False,
        summ_model_url: str = None,
        summ_model_path: str = None,
        max_doc_len: int = 5000,
        cache_size: int = 10000,
        cache_ttl: int = 3600
    ):
        """Initialize the search engine with flexible configuration."""
        self.provider = provider
        self._api_key = api_key
        self._max_results = max_results
        self._result_length = result_length
        self.process_snippets = process_snippets
        self.summ_model_url = summ_model_url
        self.summ_model_path = summ_model_path
        self._max_doc_len = max_doc_len
        
        self._memory_cache = AsyncLRUCache(cache_size, cache_ttl)
        self._setup_cache_file(cache_file)
        self._search_count = 0

    def _contains_chinese(self, text: str) -> bool:
        return any('\u4E00' <= char <= '\u9FFF' for char in text)

    def _setup_cache_file(self, cache_file: Optional[str]) -> None:
        """Set up cache file path."""
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".metis_search_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            suffix = "with_summ" if self.process_snippets else "basic"
            self._cache_file = cache_dir / f"search_{suffix}_cache.jsonl"
        else:
            self._cache_file = pathlib.Path(cache_file)
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)


    async def _load_persistent_cache(self) -> None:
        """Load cache from file asynchronously."""
        if not self._cache_file.exists():
            return
        try:
            async with aiofiles.open(self._cache_file, "r", encoding="utf-8") as f:
                cache_entries = 0
                async for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            await self._memory_cache.set(item['query'], item['result'])
                            cache_entries += 1
                        except json.JSONDecodeError:
                            continue
                print(f"Loaded {cache_entries} cache entries from {self._cache_file}")
        except Exception as e:
            print(f"Failed to load cache: {e}")

    async def _append_to_persistent_cache(self, query: str, result: Union[str, Dict]) -> None:
        """Append to persistent cache asynchronously."""
        try:
            entry = {"query": query, "result": result, "timestamp": time.time()}
            async with aiofiles.open(self._cache_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Cache write failed: {e}")

    async def _search_with_serpapi_async(self, session: aiohttp.ClientSession, query: str, timeout: int) -> Dict:
        """Performs an async search using SerpApi."""
        params = {
            "engine": "google", "q": query, "num": self._max_results, 
            "api_key": self._api_key, "google_domain": "google.com"
        }
        if self._contains_chinese(query):
            params.update({"hl": "zh-cn", "gl": "cn"})
        else:
            params.update({"hl": "en", "gl": "us"})
        
        async with session.get("https://serpapi.com/search", params=params, timeout=timeout) as response:
            response.raise_for_status()
            return await response.json()

    async def _search_with_brightdata_async(self, session: aiohttp.ClientSession, query: str, timeout: int) -> Dict:
        """Performs an async search using BrightData."""
        if not BRIGHTDATA_API_TOKEN or not BRIGHTDATA_ZONE:
            raise ValueError("Bright Data credentials are not set.")

        hl = "zh-cn" if self._contains_chinese(query) else "en"
        gl = "cn" if hl == "zh-cn" else "us"
        google_url = f"https://www.google.com/search?q={quote(query)}&hl={hl}&gl={gl}&num={self._max_results}&brd_json=1"
        
        payload = {"zone": BRIGHTDATA_ZONE, "url": google_url, "format": "raw"}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {BRIGHTDATA_API_TOKEN}"}
        
        async with session.post("https://api.brightdata.com/request", json=payload, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            return await response.json()

    async def _make_search_request(self, query: str, timeout: int) -> Dict:
        """Dispatches the search request to the configured provider."""
        async with aiohttp.ClientSession() as session:
            try:
                if self.provider == 'serpapi':
                    return await self._search_with_serpapi_async(session, query, timeout)
                elif self.provider == 'brightdata':
                    return await self._search_with_brightdata_async(session, query, timeout)
                else:
                    # Fallback to the original serper.dev logic if provider is unknown
                    # This maintains original behavior if config is not set
                    params = {"q": query, "num": self._max_results}
                    headers = {'X-API-KEY': self._api_key, 'Content-Type': 'application/json'}
                    async with session.post("https://google.serper.dev/search", json=params, headers=headers, timeout=timeout) as response:
                        response.raise_for_status()
                        return await response.json()
            except Exception as e:
                raise Exception(f"Search request via '{self.provider}' failed for '{query}': {e}") from e

    async def execute(self, query: Union[str, List[str]], timeout: int = None, prev_steps: Union[List[str], str] = None) -> str:
        """
        Executes a single or batch search with comprehensive error handling and caching.
        """
        if isinstance(query, list):
            # Handle batch query
            tasks = [self.execute(q, timeout, prev_steps) for q in query if isinstance(q, str) and q.strip()]
            if not tasks:
                return "All queries in the batch were empty."
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Format batch results
            processed_results = []
            for i, res in enumerate(results):
                if isinstance(res, Exception):
                    processed_results.append(f"Search failed for query '{query[i]}': {res}")
                else:
                    processed_results.append(res)
            return "\n\n=======\n\n".join(processed_results)

        # Handle single query (the original logic)
        query = query.strip().replace('"', '')
        if not query:
            return "Empty search query provided."
        if len(query) > 1000:
            return "Search query too long (maximum 1000 characters)."

        try:
            cached_result = await self._memory_cache.get(query)
            if cached_result is not None:
                # The logic for handling cached data with snippet processing
                if not self.process_snippets:
                    # If we cache the final string, return it
                    if isinstance(cached_result, str):
                        return cached_result
                    # If we cache raw data, re-process it
                    return await self._format_basic_results(query, cached_result)
                else:
                    data = json.loads(cached_result) if isinstance(cached_result, str) else cached_result
                    return await self._process_cached_data(query, data, prev_steps)
            
            data = await self._make_search_request(query, timeout or 45)
            result_str = await self._extract_and_format_results(query, data, prev_steps)
            
            # Cache the raw data for flexibility, or the final string for performance
            await self._cache_results(query, data if self.process_snippets else result_str)
            # await self._cache_results(query, data)
            
            return result_str
            
        except Exception as e:
            if DEBUG:
                raise e
            return f"Search failed for '{query}': {str(e)}"
    
    # ... all other methods (_process_cached_data, _cache_results, _extract_and_format_results, 
    # _format_basic_results, _process_snippets_async, etc.) remain unchanged.
    async def _process_cached_data(self, query: str, data: Dict, prev_steps: Union[List[str], str] = None) -> str:
        return await self._extract_and_format_results(query, data, prev_steps)
    
    async def _cache_results(self, query: str, data: Union[str, Dict]) -> None:
        try:
            await self._memory_cache.set(query, data)
            # Persistent cache
            cache_item = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
            await self._append_to_persistent_cache(query, cache_item)
            self._search_count += 1
        except Exception as e:
            print(f"Caching failed: {e}")
    
    async def _extract_and_format_results(self, query: str, data: Dict, prev_steps: Union[List[str], str] = None) -> str:
        # For BrightData, the results might be in a different structure
        if self.provider == 'brightdata' and isinstance(data.get("organic"), list):
            organic_results = data['organic']
        else:
            organic_results = data.get('organic', [])

        if not organic_results:
            return f"No results found for '{query}'."
        
        # We need to adapt the data structure to what the old methods expect
        # The 'organic' key is the most important one.
        adapted_data = {'organic': organic_results}

        if not self.process_snippets:
            return await self._format_basic_results(query, adapted_data)
        else:
            return await self._process_snippets_async(query, adapted_data, prev_steps)
    
    async def _format_basic_results(self, query: str, data: Dict) -> str:
        results = []
        seen_snippets = set()
        for idx, result in enumerate(data['organic'][:self._max_results], 1):
            title = result.get('title', 'No title').strip()
            link = result.get('link', '').strip()
            # Adapt to different snippet keys ('description' for brightdata, 'snippet' for others)
            snippet = result.get('snippet', result.get('description', '')).strip()
            # date_published = ""
            # if "date" in result and result["date"]:
            #     date_published = "\nDate published: " + str(result["date"])
            # source = ""
            # if "source" in result and result["source"]:
            #     source = "\nSource: " + str(result["source"])

            if snippet and snippet not in seen_snippets:
                if len(snippet) > self._result_length:
                    snippet = snippet[:self._result_length] + "..."
                # formatted = f"**{idx}.** [Title: {title}](Link: {link})\n**Snippet:** {snippet}\n"
                # formatted = f"**Page {idx}**\n**Title:** {title}\n**Link:** {link}\n**Snippet:** {snippet}\n"
                formatted = f"Page {idx}. [Title: {title}](Link: {link})\nSnippet: {snippet}\n"
                formatted = formatted.replace("Your browser can't play this video.", "")
                results.append(formatted)
                seen_snippets.add(snippet)
        return f"A Google search for '{query}' found {len(results)} results:\n\n## Web Results\n"+"\n".join(results) if results else f"No search results found for {query}."
    
    async def _process_snippets_async(self, query: str, data: Dict, prev_steps: Union[List[str], str] = None) -> str:
        # This method and its helpers (_process_single_url, _run_summarization) remain unchanged
        # as they operate on the extracted data, which we have adapted.
        max_doc_len = self._max_doc_len if self.summ_model_url else self._result_length
        do_summarization = self.summ_model_url is not None and self.summ_model_path is not None
        
        loop = asyncio.get_event_loop()
        extracted_info = await loop.run_in_executor(None, extract_relevant_info_serper, data)
        
        processing_tasks = [self._process_single_url(info, max_doc_len) for info in extracted_info]
        processed_info = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        valid_info = []
        for i, result in enumerate(processed_info):
            if isinstance(result, Exception):
                print(f"URL processing failed: {result}")
                valid_info.append(extracted_info[i])
            else:
                valid_info.append(result)
        
        formatted_document = ""
        for i, doc_info in enumerate(valid_info):
            formatted_document += f"**Web Page {i + 1}:**\n{json.dumps(doc_info, ensure_ascii=False, indent=2)}\n"

        if do_summarization and formatted_document:
            return await loop.run_in_executor(None, self._run_summarization, query, formatted_document, prev_steps)
        else:
            return formatted_document if formatted_document else "No relevant information found."
    
    async def _process_single_url(self, info: Dict, max_doc_len: int) -> Dict:
        try:
            loop = asyncio.get_event_loop()
            full_text = await loop.run_in_executor(None, lambda: extract_text_from_url(info['url'], use_jina=False))
            if full_text and not full_text.startswith("Error"):
                success, context = extract_snippet_with_context(full_text, info.get('snippet', ''), context_chars=max_doc_len)
                info['context'] = context if success else f"Could not extract context. First {max_doc_len} chars: {full_text[:max_doc_len]}"
            else:
                info['context'] = f"Failed to fetch content: {full_text or 'Unknown error'}"
        except Exception as e:
            info['context'] = f"Error processing URL: {str(e)}"
        return info
    
    def _run_summarization(self, query: str, formatted_document: str, prev_steps: Union[List[str], str] = None) -> str:
        try:
            prev_reasoning_chain = get_prev_reasoning_chain(prev_steps, begin_search_tag="<search>", begin_search_result_tag="<result>")
            return generate_webpage_to_reasonchain(prev_reasoning_chain, query, formatted_document, summ_model_url=self.summ_model_url, summ_model_path=self.summ_model_path)
        except Exception as e:
            if DEBUG: raise e
            print(f"Summarization failed: {e}")
            return formatted_document


class TextSearchHelper:
    def __init__(self, **kwargs):
        api_key = SERPER_API_KEY
        self.search_engine = GoogleSearchEngine(api_key=api_key, **kwargs)
        self._initialized = False
        self._init_lock = None

    async def _lazy_init(self):
        if not self._initialized:
            if self._init_lock is None:
                self._init_lock = asyncio.Lock()
            async with self._init_lock:
                if not self._initialized:
                    await self.search_engine._load_persistent_cache()
                    self._initialized = True

    def search(self, query: Union[str, List[str]], timeout: int = 45, prev_steps: Optional[List[str]] = None) -> Tuple[str, bool]:
        """Handles asyncio.run conflicts when an event loop is already running."""
        async def _search_task():
            await self._lazy_init()
            return await self.search_engine.execute(query, timeout, prev_steps)

        try:
            # Check if current thread has a running event loop (for Jupyter/Ray environments)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # If loop is running, execute in a separate thread to avoid deadlock
                result = self._run_in_new_thread(_search_task, timeout)
                if isinstance(result, tuple): return result
                return result, not result.startswith("Error")
            else:
                result = asyncio.run(_search_task())
        except Exception as e:
            logger.error(f"Search unexpected error: {e}")
            return f"Error: {str(e)}", False

        success = not result.startswith("Error")
        return result, success

    def _run_in_new_thread(self, coro_task, timeout: int) -> Tuple[str, bool]:
        """Run async task in an isolated thread."""
        result_container = {}
        def target():
            try:
                result_container['res'] = asyncio.run(coro_task())
            except Exception as e:
                result_container['err'] = str(e)

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeout + 10)
        
        if 'err' in result_container:
            return f"Error: {result_container['err']}", False
        if 'res' not in result_container:
            return "Error: Search thread timed out.", False
        
        res = result_container['res']
        return res, not res.startswith("Error")

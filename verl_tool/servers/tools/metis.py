from .base import BaseTool, register_tool
import regex as re
import json
from typing import Tuple, Dict, Any, Optional
import subprocess
import time
import requests
import atexit
import socket
import logging
import ray
from .utils.ipython_executor import call_python_script_with_ipython, remove_kernel

import os
from math import ceil
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64
from .utils.search_engine import TextSearchHelper
# import autopep8 

logger = logging.getLogger(__name__)

# Timeout for code execution in seconds
TIMEOUT = 20
PRE_IMPORT_LIBS = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n\n"

import black
import textwrap

def is_valid_python_code(code: str) -> Tuple[bool, str]:
    """
    Check if the code is syntactically valid Python.
    Only rejects code that is clearly not Python (like pure natural language or fatal syntax errors).
    
    Note: We normalize indentation before checking, since code may have leading indents
    that are valid in context (e.g., when inserted into an existing code block).
    
    Returns:
        (is_valid, error_message): Tuple of validation result and error message if invalid
    """
    if not code or not code.strip():
        return False, "Code is empty"
    
    # Quick heuristic: if code looks like natural language (no Python keywords/symbols)
    # This catches cases like "In the next step, I will..."
    python_indicators = ['=', 'import', 'def ', 'class ', 'if ', 'for ', 'while ', 'print', 'return', '[', '{', '(', '#']
    if not any(indicator in code for indicator in python_indicators):
        return False, "Code appears to be natural language, not Python code"
    
    # Normalize indentation before syntax check
    # This handles cases where code has leading indents (which are valid in context)
    try:
        import textwrap
        normalized_code = textwrap.dedent(code)
    except:
        normalized_code = code
    
    # Try to compile the normalized code to check for FATAL syntax errors only
    try:
        compile(normalized_code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        # Only reject truly fatal syntax errors
        # Indentation issues are already handled by normalization above
        fatal_errors = ['unterminated', 'EOF', 'invalid syntax', 'invalid character']
        if any(err in e.msg.lower() for err in fatal_errors):
            return False, f"SyntaxError: {e.msg} at line {e.lineno}"
        else:
            # Other errors are likely fixable, let them through
            return True, ""
    except Exception as e:
        return False, f"Compilation error: {str(e)}"


def fix_python_code(code: str) -> str:
    """
    Fixes Python code formatting using a two-step process:
    1. Removes common leading whitespace from the entire block (dedent).
    2. Formats the code using the 'black' formatter to fix internal indentation and style.
    
    Returns the best-effort formatted code.
    """
    try:
        dedented_code = textwrap.dedent(code).strip()
        if not dedented_code:
            return ""
        formatted_code = black.format_str(dedented_code, mode=black.Mode())
        return formatted_code

    except black.NothingChanged: 
        return dedented_code
    except Exception as e: 
        return textwrap.dedent(code).strip()

def check_forbidden_imports(code: str) -> bool:
    """
    Checks if the code contains imports of potentially dangerous packages.
    
    Args:
        code: Python code string to analyze
        
    Returns:
        Boolean indicating if the code contains forbidden imports
    """
    # List of potentially dangerous modules that could affect the host system
    forbidden_modules = [
        'subprocess', 'multiprocessing', 'threading',
        'socket', 'psutil', 'resource', 'ctypes'
    ]
    
    # Simple string-based check for import statements
    for module in forbidden_modules:
        if f"import {module}" in code or f"from {module}" in code:
            return True
    
    # Check for os.system, os.popen, and similar dangerous calls
    dangerous_patterns = [
        "os.system", "os.popen", "os.spawn", "os.fork", 
        "os.exec", "sys.exit", "os._exit", "os.kill"
    ]
    
    for pattern in dangerous_patterns:
        if pattern in code:
            return True
    
    return False


def maybe_resize_image(image):
    """
    Qwen-VL raises an error for images with height or width less than 32 pixels.
    """
    height, width = image.height, image.width
    if max(height, width) / min(height, width) > 200:
        max_val = max(height, width)
        min_val = min(height, width)

        old_scale = max_val / min_val

        max_ratio = min(150, old_scale / 2)
        target_max = int(min_val * max_ratio)

        if height > width:
            # Height is the longer side, reduce to target_max
            new_height = target_max
            new_width = int(width * old_scale / max_ratio)
        else:
            # Width is the longer side, reduce to target_max
            new_width = target_max
            new_height = int(height * old_scale / max_ratio)
        
        image = image.resize((int(new_width), int(new_height)), Image.LANCZOS)
        height, width = image.height, image.width

    if min(height, width) >= 32:
        return image

    ratio = 32 / min(height, width)
    new_height = ceil(height * ratio)
    new_width = ceil(width * ratio)
    new_image = image.resize((new_width, new_height), Image.LANCZOS)

    return new_image


def find_free_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """
    Find a free port starting from start_port.
    
    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try
        
    Returns:
        Free port number
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port after {max_attempts} attempts")

def pil_image_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

def base64_to_pil_image(base64_string: str) -> Image.Image:
    img_base64_string = base64_string.split(',', 1)[1] if 'base64,' in base64_string else base64_string
    image_data = base64.b64decode(img_base64_string)
    image = Image.open(BytesIO(image_data))
    return image

def encode_image_to_base64(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


@register_tool
class MetisTool(BaseTool):
    tool_type = "metis"
    timeout = TIMEOUT
    stop_tokens = ["</tool_call>"]
    enable_history_code_execution = False
    done_without_error = False
    pre_import_lib = False
    session_base_dir = Path(os.environ.get("METIS_SESSION_DIR", "/tmp/metis_sessions"))
    base_image_search_cache_path = os.environ.get("METIS_IMAGE_SEARCH_CACHE", "")
    image_search_cache_json_path_list = []
    
    def __init__(self, server_host: str = "127.0.0.1", server_port: int = None, **kwargs):
        """
        Args:
            server_host: Host for the IPython server
            server_port: Port for the IPython server (None = auto-select)
            **kwargs: Additional arguments passed to BaseTool
        """
        super().__init__(**kwargs)
        self.session_base_dir.mkdir(parents=True, exist_ok=True)
        self.image_search_cache_json = {}
        for cache_json_path in self.image_search_cache_json_path_list:
            if os.path.exists(cache_json_path):
                with open(cache_json_path, 'r') as f:
                    self.image_search_cache_json.update(json.load(f))
        
        self.search_engine = TextSearchHelper()

        self.tool_handlers = {
            "python": self._handle_python_execution,
            "search": self._handle_text_search,
            "text_search": self._handle_text_search,
            "image_search": self._handle_image_search,
        }
        self.action_parser = {
            "python": self._code_parse,
            "search": self._query_parse,
            "text_search": self._query_parse,
            "image_search": self._image_query_parse,
        }

    def get_usage_inst(self):
        return "You are able to write and execute Python code using Jupyter with persistent state across executions."
    
    def has_env(self, trajectory_id):
        """
        Check if the environment for the given trajectory_id exists
        """
        return trajectory_id in self.env_cache
    
    def load_env(self, trajectory_id):
        """
        Load the environment for the given trajectory_id
        """
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {
                    "turns": 0,
                    "init_code": False,
                },
                "previous_obs": [],
            }
        
        return env
    
    def save_env(self, trajectory_id, env):
        """
        Save the environment for the given trajectory_id
        """
        self.env_cache[trajectory_id] = env
    
    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation, **kwargs):
        """
        Update the environment for the given trajectory_id
        """
        env["metadata"]["turns"] += 1
        env["previous_obs"].append({
            "action": action,
            "is_valid": is_valid,
            "observation": observation,
            "extra_field": extra_field,
            **kwargs
        })
    
    def delete_env(self, trajectory_id):
        """
        Delete the environment for the given trajectory_id
        """
        if trajectory_id in self.env_cache:
            del self.env_cache[trajectory_id]
        
        session_dir = self.session_base_dir / trajectory_id
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
        
        remove_kernel(trajectory_id)

    def parse_action(self, action: str) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Parses the raw action string to extract the tool call JSON.
        
        Args:
            action: Raw action string from the model.
            
        Returns:
            A tuple containing the parsed JSON dictionary and a validity flag.
            Returns (None, False) if parsing fails or the tool is unknown.
        """
        try:
            tool_call_match = re.findall(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)
            if not tool_call_match:
                return "", False
            tool_call_content = tool_call_match[-1].strip()
            call_json = json.loads(tool_call_content)
        except:
            return "", False
        
        return call_json, True

    def conduct_action(self, trajectory_id: str, action: str, extra_field: Dict) -> Tuple[Dict, bool, bool]:
        """        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        env = self.load_env(trajectory_id)
        parsed_json, is_valid = self.parse_action(action)
        
        observation = ""
        execution_result = ""
        parsed_action = parsed_json
        done = False
        valid = False

        if is_valid:
            tool_name = parsed_json.get("name")
            arguments = parsed_json.get("arguments", {})
            handler = self.tool_handlers.get(tool_name)
            parser = self.action_parser.get(tool_name)

            if handler:
                parsed_action = parser(arguments)
                
                # --- Repeated tool call detection ---
                # Track per-tool call counts to prevent infinite loops
                tool_counts = env["metadata"].setdefault("tool_call_counts", {})
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                
                # image_search always returns the same result (query comes from extra_field, not model args)
                # so any call after the first is wasteful
                max_calls = {"image_search": 1}
                limit = max_calls.get(tool_name)
                if limit is not None and tool_counts[tool_name] > limit:
                    execution_result = (
                        f"You have already called '{tool_name}' {limit} time(s) and the results are the same. "
                        f"Do not call this tool again. "
                        f"Please analyze the results you already have and provide your final answer using <answer> tags."
                    )
                    observation = {"obs": execution_result, "image": [], "metrics": {"deduplicated": True}, "timeout": False}
                    done = False
                    valid = True
                    logger.info(f"[Metis Tool] Dedup: blocked repeated '{tool_name}' call (count={tool_counts[tool_name]}) for trajectory {trajectory_id}")
                    self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, execution_result)
                    self.save_env(trajectory_id, env)
                    return observation, done, valid
                # --- End repeated tool call detection ---
                
                observation, done, valid, execution_result = handler(trajectory_id, parsed_action, extra_field, env)
                
                # Only log if execution failed (valid=False indicates failure)
                if not valid:
                    logger.warning(
                        f"[Metis Tool] Tool execution failed\n"
                        f"  Tool: {tool_name}\n"
                        f"  Trajectory: {trajectory_id}\n"
                        f"  Result: {execution_result[:200] if execution_result else 'No output'}{'...' if len(str(execution_result)) > 200 else ''}"
                    )
            else:
                # This case should be caught by parse_action, but as a safeguard:
                execution_result = f"Error: No handler found for tool '{tool_name}'."
                observation = {"obs": execution_result, "image": [], "metrics": {}, "timeout": False}
                logger.error(f"[Metis Tool] No handler found for tool '{tool_name}' (trajectory: {trajectory_id})")
        else:
            logger.warning(f"[Metis Tool] Invalid action received (trajectory: {trajectory_id}): {action[:200]}{'...' if len(action) > 200 else ''}")

        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
    def _code_parse(self, arguments: Dict) -> str:
          user_code = arguments.get("code", "").strip()
          if not user_code:
              return ""  # No code provided
          
          # Only do basic cleanup (dedent + format), no syntax validation.
          # Let the Jupyter kernel execute it directly — Python's own stderr
          # gives much better error messages (with line numbers, context, traceback)
          # than our custom validation.
          fixed_code = fix_python_code(user_code)
          return fixed_code
    
    def _query_parse(self, arguments: Dict[str, Any]) -> str:
        query = arguments.get("query", "")
        if isinstance(query, list):
            query = query[0] if query else ""
        return str(query).strip()

    def _image_query_parse(self, arguments: Dict) -> str:
            return ""

    def _prepare_code_for_execution(self, trajectory_id: str, user_code: str, extra_field: Dict, env: Dict) -> str:
        """
        Prepares the complete Python script to be executed, including setup and history.
        """
        # FIX: Check if the kernel was evicted (cache miss) even though init_code is True.
        # When a kernel is evicted by LRU and later a new kernel is created for this trajectory,
        # the env still says init_code=True but the new kernel has no state (no imports, no variables).
        # We detect this by checking if the trajectory's kernel is still in the active cache.
        if env["metadata"].get("init_code"):
            from verl_tool.servers.tools.utils.ipython_executor import _get_kernel_manager
            try:
                manager = _get_kernel_manager()
                kernel_exists = ray.get(manager.check_kernel_exists.remote(trajectory_id))
                if not kernel_exists:
                    logger.warning(
                        f"[Metis Tool] Kernel for trajectory {trajectory_id} was evicted! "
                        f"Resetting init_code to re-run initialization and replay history."
                    )
                    env["metadata"]["init_code"] = False
                    env["metadata"]["_kernel_recovered"] = True
            except Exception as e:
                logger.warning(f"[Metis Tool] Failed to check kernel existence for {trajectory_id}: {e}. Resetting init_code.")
                env["metadata"]["init_code"] = False

        if not env["metadata"].get("init_code"):
            session_dir = self.session_base_dir / trajectory_id
            session_dir.mkdir(exist_ok=True)
            
            init_code_lines = [
                "import os",
                "import base64",
                "from PIL import Image",
                "from io import BytesIO",
                "import matplotlib.pyplot as plt",
                "import math",
                "import numpy as np",
                "from math import sin, cos",
                f"os.chdir(r'{session_dir.resolve()}')",
            ]

            images_data = extra_field.get("images", []) if extra_field else []
            
            for i, img_data in enumerate(images_data):
                try:
                    img_base64_string = img_data.split(',', 1)[1] if 'base64,' in img_data else img_data
                    init_code_lines.append(f"image_{i+1} = Image.open(BytesIO(base64.b64decode('{img_base64_string}')))")
                except Exception as e:
                    logger.warning(f"Could not process input image {i} for {trajectory_id}: {e}")

            init_code_to_execute = "\n".join(init_code_lines)
            _, success, _ = call_python_script_with_ipython(
                request_id=trajectory_id,
                script=init_code_to_execute,
                timeout=self.timeout,
            )
            if success:
                env["metadata"]["init_code"] = True

        # Prepend historical code if enabled, OR if kernel was just recovered from eviction
        # When a kernel is evicted and re-created, all intermediate variables from previous turns
        # are lost (e.g. result = image_1.crop(...)). We must replay previous code to restore them.
        kernel_was_recovered = env["metadata"].get("_kernel_recovered", False)
        if kernel_was_recovered:
            env["metadata"]["_kernel_recovered"] = False  # reset flag

        if self.enable_history_code_execution or kernel_was_recovered:
            previous_parsed_code = [obs["action"] for obs in env["previous_obs"] if obs["is_valid"]]
            if previous_parsed_code:
                if kernel_was_recovered:
                    logger.info(
                        f"[Metis Tool] Replaying {len(previous_parsed_code)} previous code blocks "
                        f"for trajectory {trajectory_id} after kernel recovery."
                    )
                code_to_execute = "\n".join(previous_parsed_code + [user_code])
            else:
                code_to_execute = user_code
        else:
            code_to_execute = user_code
            
        return code_to_execute

    def _handle_python_execution(self, trajectory_id: str, user_code: Dict, extra_field: Dict, env: Dict) -> Tuple[Dict, bool, bool, str]:
        """
        Handles the execution of Python code. This contains the logic from the original `conduct_action`.
        """
        if not user_code:
            execution_result = "Error: No code provided in the tool call arguments."
            observation = {
                "obs": execution_result,
                "image": [],
                "metrics": {"code_success": False, "code_lines": 0},
                "timeout": False,
            }
            return observation, False, False, execution_result

        code_to_execute = self._prepare_code_for_execution(trajectory_id, user_code, extra_field, env)

        if check_forbidden_imports(code_to_execute):
            execution_result = "Execution blocked: Code contains potentially dangerous operations or imports."
            logger.warning(
                f"[Metis Tool] Execution blocked for trajectory {trajectory_id}\n"
                f"  Reason: Forbidden imports detected\n"
                f"  Code preview: {user_code[:200]}{'...' if len(user_code) > 200 else ''}"
            )
            has_error = True
            img_list = []
        else:
            stdout, success, img_list = call_python_script_with_ipython(
                request_id=trajectory_id,
                script=code_to_execute,
                timeout=self.timeout,
            )
            has_error = not success
            execution_result = stdout
            
            # Only log if execution failed
            if has_error:
                logger.warning(
                    f"[Metis Tool] Code execution failed for trajectory {trajectory_id}\n"
                    f"  Code lines: {user_code.count(chr(10)) + 1}\n"
                    f"  Error output preview: {execution_result[:1000]}{'...' if len(execution_result) > 1000 else ''}"
                )

        execution_result = execution_result.lstrip(' \n')

        # FIX: Remove any <image> tags that might be in stdout to avoid conflicts
        # The agent loop will add the correct number of <image> tags based on img_list
        if "<image>" in execution_result:
            num_user_tags = execution_result.count("<image>")
            logger.warning(
                f"[Metis Tool] Found {num_user_tags} <image> tag(s) in stdout for trajectory {trajectory_id}. "
                f"These will be removed to avoid conflicts with actual image count ({len(img_list)})."
            )
            execution_result = execution_result.replace("<image>", "")

        # Build the final observation dictionary
        observation = {
            "obs": execution_result, 
            "image": img_list, 
            "metrics": {"code_success": not has_error, "code_lines": user_code.count('\n') + 1},
            "timeout": "timeout" in execution_result.lower(),
        }
        
        if self.done_without_error:
            if has_error:
                done = False
            else:
                done = True
        else: 
            done = False
            
        valid = True
        
        return observation, done, valid, execution_result
    
    def _handle_image_search(self, trajectory_id: str, data_idx: str, extra_field: Dict, env: Dict):

        img_list = []
        web_snippets = []

        query = extra_field.get('index', None)
        cached_data = self.image_search_cache_json.get(query, {})
        if not cached_data:
            observation = ""
            execution_result = f"No result found for image search."
            success = False
            valid = True
        else:
            tool_returned_web_title = cached_data.get('tool_returned_web_title', [])
            cached_images_path = cached_data.get('cached_images_path', [])
            try:
                for idx, (title, link) in enumerate(zip(tool_returned_web_title, cached_images_path)):
                    
                    image_path = os.path.join(self.base_image_search_cache_path, link)
                    if image_path is not None and os.path.exists(image_path):

                        with Image.open(image_path) as img:
                            img_format = img.format or 'PNG' 
                        img_str_base64 = encode_image_to_base64(image_path)
                        data_uri = f"data:image/{img_format.lower()};base64,{img_str_base64}"
                            
                        img_list.append(data_uri)
                        
                        redacted_version = f"{idx+1}. <image>\n[{title}]\n"
                        redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                        web_snippets.append(redacted_version)
                execution_result = f"A Google image search for the image found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)

                success = True
                valid = True

            except Exception as e:
                success = False
                valid = True
                execution_result = f"Error:{str(e)}. No results found for the image. Try with text search or direct output the answer."
        
        observation = {
            "obs": execution_result, 
            "image": img_list, 
            "metrics": {"search_success": success},
            "timeout": "timeout" in execution_result.lower()
        }

        if self.done_without_error:
            if success:
                done = True
            else:
                done = False
        else: 
            done = False
   
        return observation, done, valid, execution_result
    

    def _handle_text_search(self, trajectory_id: str, query: str, extra_field: Dict, env: Dict):

        if not query:
            execution_result = f"Error: Search query is empty."
            success = False
            valid = False
        else:
            timeout = extra_field.get('timeout', 20)
            execution_result, success = self.search_engine.search(query, timeout=timeout)
            valid = True

        observation = {
            "obs": execution_result, 
            "metrics": {"search_success": success},
            "timeout": "timeout" in execution_result.lower()
        }

        if self.done_without_error:
            if success:
                done = True
            else:
                done = False
        else: 
            done = False
   
        return observation, done, valid, execution_result

    @classmethod
    def shutdown_server(cls):
        """
        Class method to shutdown the shared server.
        """
        # Original implementation assumes a _server_manager attribute exists.
        # This check prevents an AttributeError if it's not set.
        if hasattr(cls, '_server_manager') and cls._server_manager:
            cls._server_manager.shutdown()
            cls._server_manager = None

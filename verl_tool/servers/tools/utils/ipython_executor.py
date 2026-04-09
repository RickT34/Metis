import asyncio
import base64
import contextlib
import gc
import json
import logging
import os
import queue
import re
import signal
import site
import sys
import threading
import time
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import psutil
import ray
from IPython.terminal.interactiveshell import (InteractiveShell,
                                               TerminalInteractiveShell)
from PIL import Image
from ray.exceptions import GetTimeoutError
from traitlets.config import Config


# --- Setup Logging and Paths ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress noisy logs from jupyter_client
logging.getLogger("jupyter_client").setLevel(logging.WARNING)

try:
    site_packages_dir = site.getsitepackages()[0]
except (IndexError, AttributeError):
    site_packages_dir = "/usr/local/lib/python3.10/dist-packages" # Fallback

# ==================== Jupyter-based Execution Engine  ====================
# --- Image Processing Config ---
import math

IMAGE_FACTOR = 32
MIN_PIXELS = 4 * 32 * 32      # 4096 pixels
MAX_PIXELS = 5120 * 32 * 32    # ~5.24M pixels (safety cap to prevent OOM)
MAX_RATIO = 200

# --- Math helper functions ---
def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor

# --- smart_resize core logic ---
def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, 
    min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Compute resized dimensions under total pixel constraints:
    1. Height/width divisible by factor (32)
    2. Total pixels within [min_pixels, max_pixels]
    3. Aspect ratio approximately preserved
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    
    return h_bar, w_bar

def process_image_with_pixel_limit(image: Image.Image) -> Image.Image:
    """
    Process image: handle extreme aspect ratios + smart_resize.

    Steps:
    1. Clamp extreme aspect ratios (>200)
    2. Apply smart_resize for pixel count and factor alignment
    """
    h, w = image.height, image.width
    
    # 1. Handle extreme aspect ratio (prevents Qwen2-VL errors)
    original_ratio = max(h, w) / min(h, w)
    if original_ratio > MAX_RATIO:
        logger.warning(f"Image has extreme aspect ratio {original_ratio:.1f}, clamping to {MAX_RATIO}")
        
        # Resize to target dimensions: keep shorter side, shrink longer side
        if h > w:
            # Height too large, reduce height
            new_h = int(w * MAX_RATIO)
            new_w = w
        else:
            # Width too large, reduce width
            new_w = int(h * MAX_RATIO)
            new_h = h
        
        # Resize to comply with aspect ratio limit
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        h, w = new_h, new_w
    
    # 2. Apply smart_resize (ensures factor alignment and pixel count limits)
    new_h, new_w = smart_resize(h, w)
    
    # 3. Final resize if dimensions changed
    if (new_h, new_w) != (h, w):
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return image

def process_base64_image_smart(base64_str: str) -> str:
    """
    Base64 wrapper: decode -> process -> re-encode in original format.
    """
    try:
        header = ""
        if "," in base64_str:
            header, base64_str = base64_str.split(",", 1)
            header += ","
        
        # Detect format
        fmt = "PNG"
        format_match = re.search(r"image/(\w+)", header)
        if format_match:
            fmt = format_match.group(1).upper()
            if fmt == "JPG": fmt = "JPEG"

        img_data = base64.b64decode(base64_str)
        with BytesIO(img_data) as bio:
            with Image.open(bio) as img:
                actual_fmt = img.format if img.format else fmt
                
                # Resize based on pixel limits
                processed_img = process_image_with_pixel_limit(img)

                # Re-save in original format without extra quality settings
                buffered = BytesIO()
                processed_img.save(buffered, format=actual_fmt)
                
                new_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                return header + new_base64
    except Exception as e:
        logger.warning(f"Smart resize failed: {e}")
        return header + base64_str

RETURN_PROMPT = """Code execution result:
stdout:
```
{stdout}
```

stderr:
```
{stderr}
```

{image}
"""

def strip_ansi(s: str) -> str:
    """Removes ANSI escape codes from a string."""
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', s)


class LocalJupyterSession:
    """
    Manages a single, long-lived Jupyter kernel process for code execution.
    This is a simplified version of Implementation 2's class, intended to be
    run inside a persistent actor.
    """

    @staticmethod
    def _drain_client_channels(client):
        """Helper method to drain all pending messages from client channels."""
        if not hasattr(client, 'channels_running') or not client.channels_running:
            return
        
        try:
            import zmq
            for channel_name in ['shell', 'iopub', 'stdin', 'control']:
                try:
                    channel = getattr(client, f'{channel_name}_channel', None)
                    if channel and hasattr(channel, 'socket'):
                        socket = channel.socket
                        if socket and not socket.closed:
                            socket.setsockopt(zmq.RCVTIMEO, 0)
                            drained = 0
                            # Increased limit to ensure thorough draining
                            while drained < 200:  # Increased from 50 to 200
                                try:
                                    socket.recv_multipart(zmq.NOBLOCK)
                                    drained += 1
                                except zmq.Again:
                                    break
                                except ValueError as e:
                                    # Handle DELIM errors during draining
                                    if "DELIM" in str(e):
                                        logger.debug(f"Skipping corrupted message during drain: {e}")
                                        drained += 1
                                        continue
                                    break
                                except Exception:
                                    break
                except Exception:
                    pass
        except Exception:
            pass

    def __init__(self, timeout: float = 120.0, max_retries: int = 5):
        try:
            from jupyter_client import BlockingKernelClient
            # Alias to avoid name conflict with our KernelManager
            from jupyter_client import KernelManager as JupyterKM
        except ImportError as exc:
            raise RuntimeError(
                "The Jupyter backend requires 'jupyter_client' to be installed."
            ) from exc

        self._default_timeout = timeout
        
        # Retry kernel startup to handle port conflicts
        last_error = None
        last_kernel_log = None
        for attempt in range(max_retries):
            km = None
            client = None
            try:
                # Each session owns its kernel
                km = JupyterKM()
                # Let Jupyter automatically select available ports
                km.start_kernel()
                
                # Check if kernel is actually alive after start
                if not km.is_alive():
                    raise RuntimeError("Kernel failed to start (process not alive after start_kernel)")
                
                client = km.blocking_client()
                client.start_channels()
                
                # Wait for kernel to be ready with more detailed error handling
                try:
                    client.wait_for_ready(timeout=60)
                except Exception as wait_error:
                    # Try to get kernel stderr for diagnostics
                    kernel_log = "No log available"
                    try:
                        # Check if kernel died and try to read its output
                        if hasattr(km, 'kernel') and km.kernel:
                            if hasattr(km.kernel, 'poll') and km.kernel.poll() is not None:
                                # Kernel process has terminated
                                if hasattr(km.kernel, 'stderr') and km.kernel.stderr:
                                    try:
                                        stderr_output = km.kernel.stderr.read()
                                        if stderr_output:
                                            kernel_log = f"Kernel stderr: {stderr_output[:1000]}"
                                    except:
                                        pass
                                exit_code = km.kernel.poll()
                                kernel_log += f" | Exit code: {exit_code}"
                    except Exception as log_error:
                        logger.debug(f"Could not read kernel log: {log_error}")
                    
                    last_kernel_log = kernel_log
                    raise RuntimeError(f"{wait_error} | {kernel_log}")
                
                # Success!
                self._km = km
                self._client = client
                logger.info(f"Jupyter kernel started successfully on attempt {attempt + 1}")
                return
            except Exception as e:
                last_error = e
                error_msg = str(e)
                # Check if this is a retryable error
                is_retryable = (
                    "Address already in use" in error_msg or 
                    "ZMQError" in str(type(e)) or
                    "Kernel died" in error_msg or
                    "kernel_info" in error_msg or
                    "process not alive" in error_msg
                )
                
                if is_retryable:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: Retryable error, retrying... ({e})")
                    # Clean up failed kernel thoroughly
                    try:
                        if client is not None:
                            try:
                                # Drain messages before stopping channels
                                LocalJupyterSession._drain_client_channels(client)
                                if client.channels_running:
                                    client.stop_channels()
                                time.sleep(0.2)
                            except:
                                pass
                        
                        if km is not None:
                            try:
                                if km.is_alive():
                                    km.shutdown_kernel(now=True)
                                    # Wait for kernel to terminate
                                    for _ in range(10):
                                        if not km.is_alive():
                                            break
                                        time.sleep(0.1)
                                    # Force kill if needed
                                    if km.is_alive():
                                        km.kill_kernel()
                                        time.sleep(0.2)
                            except:
                                pass
                    except:
                        pass
                    
                    # Minimal wait before retry
                    time.sleep(0.5 + attempt * 0.3)
                else:
                    # Non-retryable error, raise immediately
                    logger.error(f"Non-retryable error on attempt {attempt + 1}: {e}")
                    raise
        
        # All retries failed
        error_details = f"Failed to start Jupyter kernel after {max_retries} attempts. Last error: {last_error}"
        if last_kernel_log:
            error_details += f" | {last_kernel_log}"
        logger.error(error_details)
        raise RuntimeError(error_details)

    def execute(self, code: str, *, timeout: float | None = None) -> Dict:
        """Execute code in the kernel, returning a dictionary of outputs."""
        effective_timeout = timeout or self._default_timeout
        
        try:
            msg_id = self._client.execute(code, store_history=False, allow_stdin=False)
        except Exception as e:
            # Catch ZMQ errors during execute submission
            logger.error(f"Failed to submit code for execution: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Failed to submit code to kernel: {type(e).__name__}: {e}",
                "images": [],
            }

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        image_parts: list[str] = []
        success = True
        error_processed_from_iopub = False

        # Process messages from the iopub channel
        while True:
            try:
                msg = self._client.get_iopub_msg(timeout=effective_timeout)
            except queue.Empty as exc:
                stderr_parts.append("Timeout: Kernel didn't produce output in time.\n"+str(exc))
                success = False
                break
            except ValueError as e:
                # Handle "DELIM not in msg_list" and similar ZMQ message errors
                if "DELIM" in str(e) or "msg_list" in str(e):
                    logger.warning(f"ZMQ message format error (corrupted message): {e}")
                    stderr_parts.append(f"Kernel communication error: {e}\nKernel may need reset.")
                    success = False
                    break
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error getting iopub message: {e}")
                stderr_parts.append(f"Communication error: {type(e).__name__}: {e}")
                success = False
                break

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)
            elif msg_type == "error":
                success = False
                error_processed_from_iopub = True
                traceback_data = content.get("traceback")
                if traceback_data:
                    stderr_parts.append("\n".join(traceback_data))
                else:
                    ename = content.get("ename", "")
                    evalue = content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            elif msg_type in {"execute_result", "display_data"}:
                data = content.get("data", {})
                if "image/png" in data:
                    ori_data = f"data:image/png;base64,{data['image/png']}"
                    image_parts.append(process_base64_image_smart(ori_data))
                    # image_parts.append(f"data:image/png;base64,{data['image/png']}")
                elif "image/jpeg" in data:
                    ori_data = f"data:image/jpeg;base64,{data['image/jpeg']}"
                    image_parts.append(process_base64_image_smart(ori_data))
                    # image_parts.append(f"data:image/jpeg;base64,{data['image/jpeg']}")
                elif "text/plain" in data:
                    stdout_parts.append(data["text/plain"])
            elif msg_type == "status" and content["execution_state"] == "idle":
                break
        
        while True:
            try:
                reply = self._client.get_shell_msg(timeout=max(10.0, effective_timeout / 10))
            except queue.Empty as exc:
                stderr_parts.append("Timeout: kernel didn't send execution reply in time.\n" + str(exc))
                break
            except ValueError as e:
                # Handle "DELIM not in msg_list" and similar ZMQ message errors
                if "DELIM" in str(e) or "msg_list" in str(e):
                    logger.warning(f"ZMQ message format error on shell channel: {e}")
                    stderr_parts.append(f"Kernel communication error on shell channel: {e}")
                    success = False
                    break
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error getting shell message: {e}")
                stderr_parts.append(f"Shell communication error: {type(e).__name__}: {e}")
                success = False
                break

            if reply.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            reply_content = reply.get("content", {})
            if reply_content.get("status") == "error" and not error_processed_from_iopub:
                success = False
                if "traceback" in reply_content:
                    stderr_parts.extend(reply_content["traceback"])
                else:
                    ename = reply_content.get("ename", "UnknownError")
                    evalue = reply_content.get("evalue", "")
                    stderr_parts.append(f"{ename}: {evalue}".strip())
            break 


        # Combine and clean up outputs
        stdout = strip_ansi("".join(stdout_parts))
        stderr = strip_ansi("".join(stderr_parts))

        # IMPORTANT: If execution failed, discard any images generated
        # This prevents confusing "partial success" signals where code
        # generated images but then crashed. We want all-or-nothing execution.
        if not success and len(image_parts) > 0:
            logger.info(
                f"Execution failed but {len(image_parts)} image(s) were generated before error. "
                f"Discarding images to avoid partial success signals."
            )
            image_parts = []

        return {
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            "images": image_parts,
        }

    def close(self) -> None:
        """Shutdown the kernel and clean up resources."""
        # Stop client channels first to prevent any pending messages
        if hasattr(self, '_client'):
            try:
                if self._client.channels_running:
                    # First, drain any pending messages from all channels to prevent corrupt message errors
                    logger.debug("Draining pending messages from client channels...")
                    LocalJupyterSession._drain_client_channels(self._client)
                    
                    # Now stop the channels
                    self._client.stop_channels()
                    # Increased wait for channels to close completely
                    time.sleep(0.2)  # Increased from 0.1 to 0.2
            except Exception as e:
                logger.warning(f"Error stopping client channels: {e}")
            
            try:
                # Close the client's ZMQ sockets
                if hasattr(self._client, 'cleanup_connection'):
                    self._client.cleanup_connection()
            except Exception as e:
                logger.debug(f"Error cleaning up client connection: {e}")
        
        # Now shutdown the kernel
        if hasattr(self, '_km'):
            try:
                if self._km.is_alive():
                    self._km.shutdown_kernel(now=True)
                    # Wait for kernel to actually terminate
                    for _ in range(20):  # Wait up to 2 seconds
                        if not self._km.is_alive():
                            break
                        time.sleep(0.1)
                    
                    # Force kill if still alive
                    if self._km.is_alive():
                        logger.warning("Kernel still alive after shutdown, force killing...")
                        self._km.kill_kernel()
                        time.sleep(0.1)
                
                # Clean up kernel manager
                if hasattr(self._km, 'cleanup_connection'):
                    self._km.cleanup_connection()
            except Exception as e:
                logger.warning(f"Error shutting down kernel: {e}")

    def __del__(self) -> None:
        self.close()

# ==== Ray + Actor-based Kernel Management (from Implementation 1, adapted) ====

MAX_ACTORS = 1024  # Increased to reduce kernel eviction during training

def _ensure_ray_initialized():
    """Initialize Ray once."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        logger.info("Ray initialized")

_ensure_ray_initialized()

@ray.remote(num_cpus=0.25) # Give it a fraction of a CPU to help with scheduling
class JupyterActor:
    """
    A Ray actor that owns a single, persistent LocalJupyterSession.
    This replaces the old KernelActor.
    """
    def __init__(self):
        # Configure matplotlib to use a non-interactive backend
        # This MUST be done before importing pyplot
        # try:
        #     import matplotlib
        #     matplotlib.use('Agg')
        # except ImportError:
        #     pass # Matplotlib might not be installed

        self.actor_id = ray.get_runtime_context().get_actor_id()
        
        # Add a minimal random delay to avoid thundering herd when many actors start at once
        import random
        time.sleep(random.uniform(0, 0.1))  # Minimal jitter
        
        # Retry session creation with backoff
        max_init_retries = 3
        last_error = None
        for attempt in range(max_init_retries):
            try:
                self.session = LocalJupyterSession()
                logger.info(f"JupyterActor initialized with actor_id={self.actor_id}")
                return
            except Exception as e:
                last_error = e
                logger.error(
                    f"[ACTOR INIT ERROR] JupyterActor init attempt {attempt + 1}/{max_init_retries} failed\n"
                    f"  Actor ID: {self.actor_id}\n"
                    f"  Error type: {type(e).__name__}\n"
                    f"  Error: {e}"
                )
                if attempt < max_init_retries - 1:
                    # Aggressive backoff: 0.5s, 1s, 2s
                    wait_time = 0.5 * (2 ** attempt) + random.uniform(0, 0.3)
                    logger.info(f"Waiting {wait_time:.2f}s before retry...")
                    time.sleep(wait_time)
                    gc.collect()
        
        # All retries failed - this will cause the actor to fail to initialize
        error_msg = f"Failed to initialize JupyterActor {self.actor_id} after {max_init_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def get_actor_id(self) -> str:
        return self.actor_id

    def execute(self, script: str, max_output_size: int = 256 * 1024, timeout: int = 120) -> Tuple[str, bool]:
        """
        Run a script in the actor's Jupyter kernel.
        Returns a JSON string of the output and a success flag.
        """
        if not isinstance(script, str) or not script.strip():
            return "Error: script must be a non-empty string", False, []

        try:
            # The LocalJupyterSession handles its own timeout on message passing
            output_dict = self.session.execute(script, timeout=timeout)
            
            # Determine success based on whether stderr was produced
            success = output_dict.get("success")
            
            # Check if kernel communication error occurred
            stderr = output_dict.get("stderr", "")
            if "Kernel communication error" in stderr or "DELIM not in msg_list" in stderr:
                logger.warning(
                    f"[KERNEL CORRUPTION] ZMQ communication error detected in actor {self.actor_id}\n"
                    f"  Error: {stderr[:500]}\n"
                    f"  Resetting kernel to recover..."
                )
                try:
                    self.reset()
                    logger.info(f"Kernel reset successful for actor {self.actor_id}")
                    # Retry the execution once after reset
                    output_dict = self.session.execute(script, timeout=timeout)
                    success = output_dict.get("success")
                    stderr = output_dict.get("stderr", "")
                except Exception as reset_error:
                    logger.error(
                        f"[KERNEL RESET FAILED] Failed to reset kernel in actor {self.actor_id}\n"
                        f"  Error: {reset_error}"
                    )
                    success = False
                    output_dict = {"stdout": "", "stderr": f"Kernel reset failed: {reset_error}", "images": []}
            
            # Provide a default message for successful executions with no visible output
            if success and not output_dict.get("stdout") and not output_dict.get("images"):
                output_dict["stdout"] = "Script executed successfully (no output)."

        except Exception as e:
            logger.error(
                f"[EXECUTION ERROR] Error in JupyterActor execution\n"
                f"  Actor ID: {self.actor_id}\n"
                f"  Script length: {len(script)} chars\n"
                f"  Script preview: {script[:200]}{'...' if len(script) > 200 else ''}\n"
                f"  Timeout: {timeout}s\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error: {e}"
            )
            success = False
            output_dict = {"stdout": "", "stderr": f"Execution error: {e}", "images": []}
        
        images = output_dict.get("images", [])
        # if success:
        #     text = output_dict.get("stdout")
        # else:
        #     text = output_dict.get("stderr")
        
        text = RETURN_PROMPT.format(
            stdout=output_dict.get('stdout', ''),
            stderr=output_dict.get('stderr', ''),
            image="Images:\n" + "<image>" * len(images) if len(images) > 0 else "",
        ).strip()

        # # Truncate output if too large
        # if len(text) > max_output_size:
        #     # A simple truncation for now
        #     text = text[:max_output_size] + "\n...[output truncated]..."

        return text, success, images
        # return output_json, success

    def reset(self):
        """Resets the state by creating a new Jupyter kernel."""
        logger.info(f"Resetting JupyterActor {self.actor_id}")
        
        # Close the old session and wait for cleanup
        try:
            self.session.close()
        except Exception as e:
            logger.warning(f"Error closing session during reset: {e}")
        
        # Delete the old session reference to ensure clean state
        try:
            del self.session
        except:
            pass
        
        # Give the old kernel process and ZMQ sockets time to fully clean up
        # This is CRITICAL to avoid "Invalid Signature" and "DELIM not in msg_list" errors
        logger.debug(f"Waiting for ZMQ cleanup (actor {self.actor_id})...")
        
        # Wait time: balance between concurrency and stability
        # 0.5s was too short and caused "DELIM not in msg_list" errors in production
        # 1.0s is more reliable for high-concurrency scenarios
        time.sleep(1.0)
        
        # Force garbage collection to clean up any lingering resources
        gc.collect()
        
        # Additional small wait after GC to ensure everything is flushed
        time.sleep(0.2)
        
        # Extra cleanup: ensure all ZMQ contexts are closed
        try:
            import zmq
            # Give ZMQ time to terminate contexts
            time.sleep(0.2)
        except ImportError:
            pass
        
        # Retry creating new session with exponential backoff
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                self.session = LocalJupyterSession()
                logger.info(f"JupyterActor {self.actor_id} reset successfully")
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[RESET ERROR] Reset attempt {attempt + 1}/{max_retries} failed\n"
                    f"  Actor ID: {self.actor_id}\n"
                    f"  Error type: {type(e).__name__}\n"
                    f"  Error: {e}"
                )
                if attempt < max_retries - 1:
                    # Aggressive optimization: minimal retry delays
                    wait_time = 0.5 + attempt * 0.5
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    gc.collect()
        
        # All retries failed
        error_msg = f"Failed to reset JupyterActor {self.actor_id} after {max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

@ray.remote
class KernelManager:
    """
    Centralized manager for all JupyterActor instances.
    This class is largely unchanged from Implementation 1.
    """
    def __init__(self, max_actors: int = MAX_ACTORS):
        self.actor_cache: "OrderedDict[str, ray.actor.ActorHandle]" = OrderedDict()
        self.max_actors = max_actors
        self.manager_id = ray.get_runtime_context().get_actor_id()
        self.available_actors = []
        self.cache_lock = threading.RLock()
        # Track recent actor creations for rate limiting (balanced for stability)
        self.recent_creations = []  # List of (timestamp, actor_id) tuples
        self.max_concurrent_creations = 10  # Reduced from 20 to prevent ZMQ overload
        self.creation_window_seconds = 3.0  # Increased from 2.0 to 3.0 for more stability
        # Track consecutive failures to apply adaptive rate limiting
        self.consecutive_failures = 0
        self.failure_threshold = 5  # Apply stricter limits after this many failures
        logger.info(f"KernelManager initialized, max_actors={max_actors}")

    def _rate_limit_actor_creation(self):
        """Apply adaptive rate limiting based on system health."""
        current_time = time.time()
        
        # Clean up old entries outside the time window
        self.recent_creations = [
            (ts, aid) for ts, aid in self.recent_creations 
            if current_time - ts < self.creation_window_seconds
        ]
        
        # Apply adaptive rate limiting based on consecutive failures
        # If system is healthy (few failures), allow high concurrency
        # If system is struggling (many failures), apply stricter limits
        if self.consecutive_failures >= self.failure_threshold:
            # System is struggling, apply stricter rate limiting
            effective_limit = max(5, self.max_concurrent_creations // 4)
            logger.debug(f"System struggling ({self.consecutive_failures} failures), using strict limit: {effective_limit}")
        else:
            # System is healthy, use normal limit
            effective_limit = self.max_concurrent_creations
        
        # Check if we're at the limit
        if len(self.recent_creations) >= effective_limit:
            oldest_time = self.recent_creations[0][0]
            wait_time = self.creation_window_seconds - (current_time - oldest_time)
            if wait_time > 0:
                logger.warning(
                    f"Rate limit: {len(self.recent_creations)} actors created in last "
                    f"{self.creation_window_seconds}s. Waiting {wait_time:.2f}s... "
                    f"(failures: {self.consecutive_failures})"
                )
                time.sleep(wait_time + 0.05)  # Reduced buffer from 0.1 to 0.05
                # Clean up again after waiting
                current_time = time.time()
                self.recent_creations = [
                    (ts, aid) for ts, aid in self.recent_creations 
                    if current_time - ts < self.creation_window_seconds
                ]
    
    def _record_actor_creation(self, actor_id: str):
        """Record that an actor was created for rate limiting purposes."""
        self.recent_creations.append((time.time(), actor_id))

    def get_or_create_actor(self, request_id: str) -> "ray.actor.ActorHandle":
        """Get existing actor or create/recycle one for the given request_id."""
        with self.cache_lock:
            if request_id in self.actor_cache:
                actor = self.actor_cache[request_id]
                self.actor_cache.move_to_end(request_id) # Mark as recently used
                return actor
            
            logger.info(f"Cache MISS for request_id={request_id}. Looking for an actor...")
            
            if self.available_actors:
                # Recycle an available actor
                actor = self.available_actors.pop()
                logger.info(f"Recycling available actor for {request_id}")
                try:
                    ray.get(actor.reset.remote())
                    # Reset successful, decrease failure counter
                    self.consecutive_failures = max(0, self.consecutive_failures - 1)
                except Exception as e:
                    logger.error(
                        f"[MANAGER ERROR] Failed to reset recycled actor\n"
                        f"  Request ID: {request_id}\n"
                        f"  Consecutive failures: {self.consecutive_failures + 1}\n"
                        f"  Error type: {type(e).__name__}\n"
                        f"  Error: {e}\n"
                        f"  Creating new actor instead..."
                    )
                    self.consecutive_failures += 1
                    # Kill the failed actor and create a new one
                    try:
                        ray.kill(actor)
                    except:
                        pass
                    self._rate_limit_actor_creation()
                    try:
                        actor = JupyterActor.remote()
                        self._record_actor_creation(request_id)
                        self.consecutive_failures = max(0, self.consecutive_failures - 1)
                    except Exception as create_error:
                        logger.error(f"Failed to create new actor: {create_error}")
                        self.consecutive_failures += 1
                        raise
            elif len(self.actor_cache) + len(self.available_actors) < self.max_actors:
                # Create a new actor if we are under the limit
                logger.info(f"Creating new JupyterActor for {request_id}")
                self._rate_limit_actor_creation()
                try:
                    actor = JupyterActor.remote()
                    self._record_actor_creation(request_id)
                    # Creation successful, decrease failure counter
                    self.consecutive_failures = max(0, self.consecutive_failures - 1)
                except Exception as e:
                    logger.error(
                        f"[MANAGER ERROR] Failed to create new actor\n"
                        f"  Request ID: {request_id}\n"
                        f"  Consecutive failures: {self.consecutive_failures + 1}\n"
                        f"  Error type: {type(e).__name__}\n"
                        f"  Error: {e}"
                    )
                    self.consecutive_failures += 1
                    raise
            else:
                # Evict the least recently used actor if the cache is full
                lru_id, lru_actor = self.actor_cache.popitem(last=False)
                logger.warning(f"Cache is full. Evicting LRU actor for session '{lru_id}' to make room for '{request_id}'.")
                actor = lru_actor
                try:
                    ray.get(actor.reset.remote())
                    self.consecutive_failures = max(0, self.consecutive_failures - 1)
                except Exception as e:
                    logger.error(
                        f"[MANAGER ERROR] Failed to reset evicted actor\n"
                        f"  Request ID: {request_id}\n"
                        f"  LRU ID: {lru_id}\n"
                        f"  Consecutive failures: {self.consecutive_failures + 1}\n"
                        f"  Error type: {type(e).__name__}\n"
                        f"  Error: {e}\n"
                        f"  Creating new actor instead..."
                    )
                    self.consecutive_failures += 1
                    # Kill the failed actor and create a new one
                    try:
                        ray.kill(actor)
                    except:
                        pass
                    self._rate_limit_actor_creation()
                    try:
                        actor = JupyterActor.remote()
                        self._record_actor_creation(request_id)
                        self.consecutive_failures = max(0, self.consecutive_failures - 1)
                    except Exception as create_error:
                        logger.error(
                            f"[MANAGER ERROR] Failed to create new actor (after eviction failure)\n"
                            f"  Request ID: {request_id}\n"
                            f"  Consecutive failures: {self.consecutive_failures + 1}\n"
                            f"  Error type: {type(create_error).__name__}\n"
                            f"  Error: {create_error}"
                        )
                        self.consecutive_failures += 1
                        raise
            
            self.actor_cache[request_id] = actor
            return actor
    
    def get_stats(self) -> Dict:
        """Get statistics about the kernel manager and system health."""
        current_time = time.time()
        recent_creations_count = len([
            ts for ts, _ in self.recent_creations 
            if current_time - ts < self.creation_window_seconds
        ])
        
        # Determine health status based on consecutive failures
        if self.consecutive_failures == 0:
            health = "healthy"
        elif self.consecutive_failures < self.failure_threshold:
            health = "degraded"
        else:
            health = "unhealthy"
        
        return {
            "active_sessions": len(self.actor_cache),
            "available_actors": len(self.available_actors),
            "total_actors": len(self.actor_cache) + len(self.available_actors),
            "max_actors": self.max_actors,
            "session_ids": list(self.actor_cache.keys()),
            "health": health,
            "consecutive_failures": self.consecutive_failures,
            "recent_creations": recent_creations_count,
            "rate_limit_window": f"{self.creation_window_seconds}s",
            "rate_limit_max": self.max_concurrent_creations,
        }

    def check_kernel_exists(self, request_id: str) -> bool:
        """Check if a kernel for the given request_id is still in the active cache."""
        with self.cache_lock:
            return request_id in self.actor_cache

    def remove_kernel(self, request_id: str) -> bool:
        """Moves a kernel from the active cache to the available pool."""
        with self.cache_lock:
            if request_id in self.actor_cache:
                actor = self.actor_cache.pop(request_id)
                self.available_actors.append(actor)
                logger.info(f"Kernel for request_id={request_id} released to available pool.")
                return True
            return False

    def cleanup_all(self) -> None:
        """Terminates all actors and clears all caches."""
        with self.cache_lock:
            all_actors = list(self.actor_cache.values()) + self.available_actors
            logger.info(f"Cleaning up {len(all_actors)} actors.")
            for actor in all_actors:
                ray.kill(actor)
            self.actor_cache.clear()
            self.available_actors.clear()
        gc.collect()

# --- Global Singleton Management ---
_kernel_manager: Optional[ray.actor.ActorHandle] = None
_manager_lock = threading.Lock()

def _get_kernel_manager() -> "ray.actor.ActorHandle":
    """Get or create the singleton KernelManager actor."""
    global _kernel_manager
    with _manager_lock:
        if _kernel_manager is not None:
            return _kernel_manager
        
        try:
            _kernel_manager = ray.get_actor("kernel_manager")
            logger.info("Found existing KernelManager actor.")
        except ValueError:
            logger.info("Creating new KernelManager actor.")
            _kernel_manager = KernelManager.options(
                name="kernel_manager",
                lifetime="detached",
            ).remote(max_actors=MAX_ACTORS)
        
        return _kernel_manager

# --- Public API Functions ---

async def call_python_async(
    request_id: str,
    script: str,
    timeout: int = 120,
    max_output_size: int = 256 * 1024,
) -> Tuple[str, bool]:
    """
    Asynchronously execute a Python script using a managed Jupyter actor.
    The output is a JSON string containing stdout, stderr, and images.
    """
    if not isinstance(request_id, str) or not request_id.strip():
        return "Error: request_id must be a non-empty string", False, []

    manager = _get_kernel_manager()
    actor = await manager.get_or_create_actor.remote(request_id)
    
    try:
        # Use ray.get with a timeout as a hard stop
        future = actor.execute.remote(script, max_output_size, timeout)
        output, success, image = await asyncio.to_thread(ray.get, future, timeout=(timeout + 5))
        return output, success, image
    except GetTimeoutError:
        logger.warning(
            f"[RAY TIMEOUT] Actor unresponsive, killing it\n"
            f"  Request ID: {request_id}\n"
            f"  Timeout: {timeout}s\n"
            f"  Script length: {len(script)} chars\n"
            f"  Actor will be removed from cache and recreated on next request"
        )
        # The actor is unresponsive, kill it. The manager will create a new one next time.
        ray.kill(actor)
        # Manually remove it from the cache to be certain
        await manager.remove_kernel.remote(request_id) 
        return f"Error: Execution timed out after {timeout} seconds and the kernel was terminated.", False, []
    except Exception as e:
        logger.error(
            f"[ORCHESTRATION ERROR] Unexpected error during remote call\n"
            f"  Request ID: {request_id}\n"
            f"  Script length: {len(script)} chars\n"
            f"  Timeout: {timeout}s\n"
            f"  Error type: {type(e).__name__}\n"
            f"  Error: {e}"
        )
        return f"Error: Orchestration error: {str(e)}", False, []


def call_python_script_with_ipython(
    request_id: str,
    script: str,
    timeout: int = 120,
    max_output_size: int = 256 * 1024,
) -> Tuple[str, bool]:
    """Synchronous wrapper for call_python_async."""
    return asyncio.run(
        call_python_async(request_id, script, timeout, max_output_size)
    )

def get_kernel_stats() -> Dict:
    """Get statistics about the kernel manager."""
    manager = _get_kernel_manager()
    stats = ray.get(manager.get_stats.remote())
    stats["process_memory_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
    return stats

def remove_kernel(request_id: str) -> None:
    """Releases a kernel back to the available pool."""
    manager = _get_kernel_manager()
    ray.get(manager.remove_kernel.remote(request_id))

def cleanup_all_kernels() -> None:
    """Terminates all running kernels."""
    manager = _get_kernel_manager()
    ray.get(manager.cleanup_all.remote())

def _get_actor(request_id: str) -> "ray.actor.ActorHandle":
    """
    Get the KernelActor for the given request_id via the manager.
    """
    manager = _get_kernel_manager()
    actor = ray.get(manager.get_or_create_actor.remote(request_id))
    return actor


# ==============================================================================
#  Example Usage
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("\n--- Testing Hybrid Ray + Jupyter Kernel Executor ---\n")
    
    SESSION_ID = "user_session_123"

    # 1. Simple execution
    print("--- 1. Simple print statement ---")
    output_json, success = call_python(SESSION_ID, "print('Hello, Jupyter!')")
    print(f"Success: {success}")
    print(f"Output: {json.loads(output_json)}")
    assert success and "Hello, Jupyter!" in json.loads(output_json)['stdout']

    # 2. State persistence test
    print("\n--- 2. State persistence test ---")
    call_python(SESSION_ID, "x = 100")
    output_json, success = call_python(SESSION_ID, "print(f'The value of x is {x+1}')")
    print(f"Success: {success}")
    print(f"Output: {json.loads(output_json)}")
    assert success and "The value of x is 101" in json.loads(output_json)['stdout']

    # 3. Rich output (Image) test
    print("\n--- 3. Rich output (matplotlib plot) test ---")
    plot_script = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(5, 3))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()
"""
    output_json, success = call_python(SESSION_ID, plot_script, timeout=20)
    output_data = json.loads(output_json)
    print(f"Success: {success}")
    print(f"Stdout: {output_data['stdout']}")
    print(f"Stderr: {output_data['stderr']}")
    print(f"Number of images captured: {len(output_data['images'])}")
    if output_data['images']:
        print(f"Image 1 data (truncated): {output_data['images'][0][:80]}...")
        # You could save and open the image like this:
        # img_data = base64.b64decode(output_data['images'][0].split(',')[1])
        # with open("plot.png", "wb") as f:
        #     f.write(img_data)
        # print("Saved plot to plot.png")
    assert success and len(output_data['images']) == 1

    # 4. Timeout test
    print("\n--- 4. Timeout test ---")
    timeout_script = "import time; time.sleep(10)"
    output_json, success = call_python("timeout_session", timeout_script, timeout=3)
    print(f"Success: {success}")
    print(f"Output: {json.loads(output_json)}")
    assert not success and "timed out" in json.loads(output_json)['stderr'].lower()

    # 5. Check stats
    print("\n--- 5. Kernel Stats ---")
    stats = get_kernel_stats()
    print(json.dumps(stats, indent=2))
    assert stats['active_sessions'] >= 1

    # 6. Release a kernel
    print("\n--- 6. Releasing a kernel ---")
    remove_kernel(SESSION_ID)
    stats_after_remove = get_kernel_stats()
    print("Stats after removing session:")
    print(json.dumps(stats_after_remove, indent=2))
    assert stats_after_remove['active_sessions'] < stats['active_sessions']
    assert stats_after_remove['available_actors'] > 0

    # 7. Clean up everything
    print("\n--- 7. Cleaning up all kernels ---")
    cleanup_all_kernels()
    stats_after_cleanup = get_kernel_stats()
    print("Stats after cleanup:")
    print(json.dumps(stats_after_cleanup, indent=2))
    assert stats_after_cleanup['total_actors'] == 0
    
    print("\n--- All tests passed! ---")


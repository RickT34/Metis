from .base import BaseTool, register_tool
import regex as re
import json
from typing import Tuple
import subprocess
import time
import requests
import atexit
import socket
import logging
import os
from .utils.ipython_executor import call_python_script_with_ipython, remove_kernel

from pathlib import Path
from PIL import Image
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

# Timeout for code execution in seconds
TIMEOUT = 10
PRE_IMPORT_LIBS = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(6*10**5)\n\n"

import black
import textwrap

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

def encode_image_url(image: Image.Image) -> str:
    buffered = BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_string}"

def decode_image_url(image_url: str) -> Image.Image:
    if "base64," in image_url:
        base64_string = image_url.split('base64,')[1]
    else:
        base64_string = image_url
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

@register_tool
class MetisCodeTool(BaseTool):
    tool_type = "metis_code"
    timeout = TIMEOUT
    stop_tokens = ["<tool_call>"]
    enable_history_code_execution = False
    done_without_error = False
    pre_import_lib = False

    SESSION_BASE_DIR = Path(os.environ.get("METIS_SESSION_DIR", "/tmp/metis_sessions"))

    def __init__(self, server_host: str = "127.0.0.1", server_port: int = None, **kwargs):
        """
        
        Args:
            server_host: Host for the IPython server
            server_port: Port for the IPython server (None = auto-select)
            **kwargs: Additional arguments passed to BaseTool
        """
        super().__init__(**kwargs)
        self.SESSION_BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    def get_usage_inst(self):
        return "You are able to write and execute Python code using IPython with persistent state across executions."
    
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
        
        session_dir = self.SESSION_BASE_DIR / trajectory_id
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
        
        remove_kernel(trajectory_id)
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing Python code
            
        Returns:
            Tuple containing the extracted code and a validity flag
        """
        tool_call_match = re.findall(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)
        if not tool_call_match:
            all_valid_python_code = []

        else:
            tool_call_content = tool_call_match[-1].strip()
            call_json = json.loads(tool_call_content)
            tool_name = call_json.get("name", "").strip()
            args = call_json.get("arguments", {})
            if tool_name == "python":
                code = args.get("code", "").strip()
                if code:
                    all_valid_python_code = [code]
                else:
                    all_valid_python_code = []
        
        if len(all_valid_python_code) == 0:
            return "", False
        
        # Use all the code blocks
        parsed_code = "\n".join([code.strip() for code in all_valid_python_code])
        
        return parsed_code, True
    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed action using IPython
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = ""
            execution_result = ""
            done = False
            valid = False
        else:
            # Extract stdin if provided in extra_field
            stdin = extra_field.get("stdin", "") if extra_field else None
            
            test_input = re.findall(r"```input\n(.*?)\n```", action, re.DOTALL)
            if len(test_input) > 0:
                stdin = test_input[0].strip()
            
            session_dir = self.SESSION_BASE_DIR / trajectory_id
            session_dir.mkdir(exist_ok=True)
            init_code_lines = [
                "import os",
                "import base64",
                "from PIL import Image",
                "from io import BytesIO",
                "import cv2",
                "import numpy as np",
                f"os.chdir(r'{session_dir.resolve()}')",
            ]
            
            images_data = extra_field.get("images", []) if extra_field else []
            for i, img_data in enumerate(images_data):
                try:
                    if 'base64' in img_data:
                        img_base64_string = img_data.split(',', 1)[1]
                    else:
                        img_base64_string = img_data
                    init_code_lines.append(f"image_{i+1} = Image.open(BytesIO(base64.b64decode('{img_base64_string}')))")
                except Exception as e:
                    logger.warning(f"Could not process input image {i} for {trajectory_id}: {e}")

            code_to_execute = "\n".join(init_code_lines) + "\n\n" + parsed_action
            code_to_execute = fix_python_code(code_to_execute)
            
            # Determine what code to execute
            if self.enable_history_code_execution:
                previous_parsed_code = [obs["action"] for obs in env["previous_obs"] if obs["is_valid"]]
                code_to_execute = previous_parsed_code + [code_to_execute]
            else:
                code_to_execute = code_to_execute
            if check_forbidden_imports(code_to_execute):
                stdout = ""
                stderr = "Execution blocked: Code contains potentially dangerous operations or imports."
                has_error = True
                execution_result = stdout + "\n" + stderr
            else:
                stdout, success, img_list = call_python_script_with_ipython(
                    request_id=trajectory_id,
                    script=code_to_execute,
                    timeout=self.timeout,
                )
                has_error = not success
                execution_result = stdout
            execution_result = execution_result.lstrip(' \n')
            execution_result += "\nProcessed Images:\n" + "<image>" * len(img_list) if len(img_list) > 0 else ""
            observation = execution_result

            # Format the observation based on the action type
            if action.endswith("</tool_call>"):
                observation = observation
            else:
                observation = "\n" + observation + "\n"

            if self.done_without_error:
                if has_error:
                    done = False
                else:
                    done = True
            else: 
                done = False
            valid = True
            
            observation = {"obs": observation, "image": img_list, "metrics": {"code_success": success, "code_lines": parsed_action.count('\n') + 1}, "timeout": "execution time out" in execution_result.lower()}
        
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
    
    @classmethod
    def shutdown_server(cls):
        """
        Class method to shutdown the shared server.
        Call this when shutting down the application.
        """
        if hasattr(cls, '_server_manager') and cls._server_manager:
            cls._server_manager.shutdown()
            cls._server_manager = None
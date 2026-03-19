from __future__ import annotations

from typing import Any, Dict, List, ClassVar
import asyncio
import os
import tempfile

from base.agent.base_action import BaseAction


class ExecuteCodeAction(BaseAction):
    name: str = "ExecuteCodeAction"
    description: str = "Execute code in a sandboxed subprocess (python|bash). Default disabled; enable via config and whitelist."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "code_type": {"type": "string", "enum": ["python", "bash"], "default": "python"},
            "timeout_sec": {"type": "integer", "default": 10},
        },
        "required": ["code", "code_type"],
        "additionalProperties": False,
    }

    DISALLOWED_BASH: ClassVar[List[str]] = [
        " rm ",
        "sudo",
        "chmod",
        "chown",
        ">/",
        "> /",
        " mv /",
        " dd ",
        " mkfs",
        " mount ",
    ]

    async def __call__(self, **kwargs) -> Any:
        code = kwargs.get("code", "")
        code_type = kwargs.get("code_type", "python")
        timeout = int(kwargs.get("timeout_sec", 10))

        try:
            if code_type == "bash":
                return await self._exec_bash(code, timeout)
            elif code_type == "python":
                return await self._exec_python(code, timeout)
            else:
                return {"success": False, "output": None, "error": f"Unsupported code_type: {code_type}", "metrics": {}}
        except Exception as e:
            return {"success": False, "output": None, "error": str(e), "metrics": {}}

    async def _exec_bash(self, code: str, timeout: int) -> Dict[str, Any]:
        low = f" {code.strip()} ".lower()
        for bad in self.DISALLOWED_BASH:
            if bad in low:
                return {"success": False, "output": None, "error": f"disallowed command in bash: {bad.strip()}", "metrics": {}}
        proc = await asyncio.create_subprocess_shell(
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"success": False, "output": None, "error": "timeout"}
        ok = proc.returncode == 0
        return {"success": ok, "output": stdout.decode("utf-8", "replace"), "error": None if ok else stderr.decode("utf-8", "replace"), "metrics": {}}

    async def _exec_python(self, code: str, timeout: int) -> Dict[str, Any]:
        base = os.path.abspath(os.getcwd())
        tmpdir = os.path.join(base, "workspace", ".exec")
        os.makedirs(tmpdir, exist_ok=True)
        downloads_dir = os.path.join(base, "workspace", "tmp")
        os.makedirs(downloads_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=tmpdir, suffix=".py") as tf:
            tf.write(code)
            path = tf.name
        cmd = f"DOWNLOAD_DIR='{downloads_dir}' python '{path}'"
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"success": False, "output": None, "error": "timeout"}
        ok = proc.returncode == 0
        return {"success": ok, "output": stdout.decode("utf-8", "replace"), "error": None if ok else stderr.decode("utf-8", "replace"), "metrics": {}}

"""
core/mcp_client.py
---------------------
Real Model Context Protocol (MCP) client for Alpaca's official
`alpaca-mcp-server` package.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import shutil
import sys
import threading
from concurrent.futures import Future
from contextlib import AsyncExitStack
from typing import Any, Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ============================================================
# ENV LOADING - SEARCH IN MULTIPLE LOCATIONS
# ============================================================
def _load_env_file():
    """Find and load .env file from multiple possible locations."""
    # 1. Current working directory
    cwd_env = os.path.join(os.getcwd(), '.env')
    if os.path.exists(cwd_env):
        load_dotenv(cwd_env)
        print(f"✅ .env loaded from: {cwd_env}")
        return True
    
    # 2. Script directory (core/ klasörünün bir üstü)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # OrbiTrade klasörü
    parent_env = os.path.join(parent_dir, '.env')
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
        print(f"✅ .env loaded from: {parent_env}")
        return True
    
    # 3. Project root (bu dosyanın 2 üstü)
    grandparent_dir = os.path.dirname(parent_dir)
    grandparent_env = os.path.join(grandparent_dir, '.env')
    if os.path.exists(grandparent_env):
        load_dotenv(grandparent_env)
        print(f"✅ .env loaded from: {grandparent_env}")
        return True
    
    # 4. Fallback: default load_dotenv()
    load_dotenv()
    print("⚠️ .env loaded from default location (current directory)")
    return False

# ENV'yi yükle
_load_env_file()

# ============================================================
# DEBUG: check API keys
# ============================================================
if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_SECRET_KEY"):
    print("⚠️ ALPACA_API_KEY or ALPACA_SECRET_KEY can't be found!")
    print(f"   ALPACA_API_KEY: {'SET' if os.getenv('ALPACA_API_KEY') else 'MISSING'}")
    print(f"   ALPACA_SECRET_KEY: {'SET' if os.getenv('ALPACA_SECRET_KEY') else 'MISSING'}")
    print("   Please check the .env file.")
else:
    print(f"✅ ALPACA_API_KEY found (length: {len(os.getenv('ALPACA_API_KEY', ''))})")
    print(f"✅ ALPACA_SECRET_KEY found (length: {len(os.getenv('ALPACA_SECRET_KEY', ''))})")

_CONNECT_TIMEOUT = 30.0
_CALL_TIMEOUT = 30.0


def _resolve_server_command() -> str:
    """Locates the alpaca-mcp-server executable."""
    exe = shutil.which("alpaca-mcp-server")
    if exe:
        return exe

    candidate_dir = os.path.dirname(sys.executable)
    for name in ("alpaca-mcp-server", "alpaca-mcp-server.exe"):
        candidate = os.path.join(candidate_dir, name)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not locate 'alpaca-mcp-server'. Make sure "
        "`pip install -r requirements.txt` succeeded."
    )


def _parse_tool_result(result: Any) -> Any:
    """Parses MCP tool response."""
    if getattr(result, "isError", False):
        detail = result.content[0].text if result.content else "unknown error"
        raise RuntimeError(f"Alpaca MCP tool call failed: {detail}")

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured

    for block in result.content:
        if getattr(block, "type", None) == "text":
            try:
                return json.loads(block.text)
            except (json.JSONDecodeError, TypeError):
                return block.text

    return None


class _MCPWorker:
    """Owns the background event loop and the persistent MCP session."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run_loop, name="alpaca-mcp-session", daemon=True
        )
        self._thread.start()

        if not self._ready.wait(timeout=_CONNECT_TIMEOUT):
            raise TimeoutError("Timed out starting the Alpaca MCP server session.")
        if self._start_error is not None:
            error, self._start_error = self._start_error, None
            raise error

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect())
        except BaseException as exc:
            self._start_error = exc
            self._ready.set()
            return
        self._loop.run_forever()

    async def _connect(self) -> None:
        # ENV kontrolü (tekrar kontrol et)
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            # Tekrar dene - belki farklı bir yerde
            env_paths = [
                os.path.join(os.getcwd(), '.env'),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),
                os.path.join(os.path.dirname(__file__), '..', '.env'),
            ]
            for env_path in env_paths:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    api_key = os.getenv("ALPACA_API_KEY")
                    secret_key = os.getenv("ALPACA_SECRET_KEY")
                    if api_key and secret_key:
                        break
        
        if not api_key or not secret_key:
            raise EnvironmentError(
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are not set in .env.\n"
                f"Current directory: {os.getcwd()}\n"
                f"File exists at .env: {os.path.exists('.env')}\n"
                f"File exists at parent: {os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))}"
            )

        params = StdioServerParameters(
            command=_resolve_server_command(),
            args=[],
            env=dict(os.environ),
        )
        self._exit_stack = AsyncExitStack()
        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        self._ready.set()

    def call(self, tool_name: str, arguments: dict) -> Any:
        if self._loop is None or self._session is None:
            raise RuntimeError("MCP session is not started. Call start() first.")

        future: Future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(tool_name, arguments), self._loop
        )
        result = future.result(timeout=_CALL_TIMEOUT)
        return _parse_tool_result(result)

    def stop(self) -> None:
        if self._loop is None:
            return

        async def _close() -> None:
            if self._exit_stack is not None:
                await self._exit_stack.aclose()

        try:
            fut = asyncio.run_coroutine_threadsafe(_close(), self._loop)
            fut.result(timeout=10)
        except Exception:
            pass

        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

        self._loop = None
        self._thread = None
        self._session = None


_worker: Optional[_MCPWorker] = None
_worker_lock = threading.Lock()


def get_worker() -> _MCPWorker:
    """Returns the process-wide MCP session, starting it on first use."""
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _MCPWorker()
            _worker.start()
        return _worker


def call_tool(tool_name: str, arguments: Optional[dict] = None) -> Any:
    """Synchronous entry point: call_tool('get_account_info', {})."""
    return get_worker().call(tool_name, arguments or {})


def list_tools() -> list[str]:
    """
    Diagnostic helper: returns exact tool names your installed
    alpaca-mcp-server exposes.
    """
    worker = get_worker()
    if worker._loop is None or worker._session is None:
        raise RuntimeError("MCP session is not started.")

    future: Future = asyncio.run_coroutine_threadsafe(
        worker._session.list_tools(), worker._loop
    )
    result = future.result(timeout=_CALL_TIMEOUT)
    return sorted(tool.name for tool in result.tools)


def shutdown() -> None:
    """Stops the MCP session."""
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()
            _worker = None


atexit.register(shutdown)
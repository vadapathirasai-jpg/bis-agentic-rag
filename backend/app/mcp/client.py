"""BIS Sahayak MCP Client Module.

Provides a clean, modular client interface for invoking read-only tools on the
BIS Sahayak Model Context Protocol (MCP) server.

Strict Guarantees:
- Pure read-only consumer: invokes 'search_bis_documents'.
- Fails safely: all protocol, transport, and execution errors raise MCPClientError,
  enabling the Consumer Agent to seamlessly fall back to the direct Retriever.
- Validates that returned evidence chunks preserve full metadata.
- Thread-safe and compatible with both synchronous and asynchronous contexts.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bis_sahayak.mcp_client")


class MCPClientError(Exception):
    """Raised when an MCP client request fails or returns an invalid payload."""


class BISConsumerMCPClient:
    """Standardized MCP Client for interacting with the BIS Sahayak MCP Server."""

    def __init__(self, server: Optional[Any] = None) -> None:
        """Initialize the MCP client.

        Args:
            server: Optional MCPServer instance. If None, imports and uses the
                canonical 'mcp' instance from app.mcp.server.
        """
        if server is not None:
            self.server = server
        else:
            try:
                from app.mcp.server import mcp as default_mcp_server

                self.server = default_mcp_server
            except Exception as err:
                logger.warning("Could not load app.mcp.server: %s", err)
                self.server = None

    @property
    def is_available(self) -> bool:
        """Check whether the MCP server interface is accessible."""
        return self.server is not None and hasattr(self.server, "call_tool")

    def _execute_coroutine(self, coro: Any, timeout: float = 15.0) -> Any:
        """Safely execute an asynchronous coroutine from either sync or async context.

        Args:
            coro: The coroutine to execute.
            timeout: Maximum execution timeout in seconds.

        Returns:
            The coroutine result.

        Raises:
            MCPClientError: If execution fails or times out.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop and loop.is_running():
                # Running inside an active event loop (e.g., FastAPI async handler)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result(timeout=timeout)
            else:
                # Synchronous context (standard CLI or worker thread)
                return asyncio.run(coro)
        except concurrent.futures.TimeoutError as te:
            raise MCPClientError(f"MCP tool execution timed out after {timeout}s.") from te
        except Exception as exc:
            raise MCPClientError(f"MCP async execution error: {exc}") from exc

    def search_bis_documents(
        self,
        query: str,
        top_k: int = 5,
        timeout: float = 15.0,
    ) -> List[Dict[str, Any]]:
        """Invoke 'search_bis_documents' on the MCP server to retrieve relevant BIS evidence.

        Args:
            query: Natural language query string.
            top_k: Maximum candidate chunks to retrieve.
            timeout: Network/processing timeout in seconds.

        Returns:
            List of structured evidence chunk dictionaries.

        Raises:
            MCPClientError: If MCP is unavailable, tool execution fails, or payload is invalid.
        """
        if not self.is_available:
            raise MCPClientError("MCP server is unavailable or not configured.")

        if not query or not isinstance(query, str) or not query.strip():
            raise MCPClientError("Query must be a non-empty string.")

        arguments = {
            "query": query.strip(),
            "top_k": max(1, int(top_k)),
        }

        async def _call() -> Any:
            return await self.server.call_tool("search_bis_documents", arguments)

        logger.debug("Calling MCP tool 'search_bis_documents' with args: %s", arguments)
        try:
            call_result = self._execute_coroutine(_call(), timeout=timeout)
        except Exception as exc:
            raise MCPClientError(f"MCP call_tool failed: {exc}") from exc

        if getattr(call_result, "is_error", False):
            raise MCPClientError(f"MCP tool returned error status: {call_result}")

        # Parse output payload from structured_content or content TextContent
        parsed_payload: Optional[Dict[str, Any]] = None

        if hasattr(call_result, "structured_content") and call_result.structured_content:
            struct = call_result.structured_content
            if isinstance(struct, dict) and "result" in struct:
                parsed_payload = struct["result"]
            elif isinstance(struct, dict):
                parsed_payload = struct

        if parsed_payload is None and hasattr(call_result, "content") and call_result.content:
            for content_item in call_result.content:
                text = getattr(content_item, "text", None)
                if text:
                    try:
                        parsed_payload = json.loads(text)
                        break
                    except json.JSONDecodeError as jde:
                        logger.warning("Failed to decode JSON from MCP TextContent: %s", jde)

        if not parsed_payload or not isinstance(parsed_payload, dict):
            raise MCPClientError("MCP server returned invalid or empty response payload.")

        if not parsed_payload.get("success", False):
            err_msg = parsed_payload.get("error", "Unknown MCP tool failure.")
            raise MCPClientError(f"MCP tool indicated failure: {err_msg}")

        results = parsed_payload.get("results", [])
        if not isinstance(results, list):
            raise MCPClientError("MCP 'results' field is not a list.")

        # Defensive metadata preservation verification
        required_metadata_keys = {
            "chunk_id",
            "text",
            "title",
            "source_url",
            "category",
            "document_type",
            "score",
            "section",
        }
        for chunk in results:
            if not isinstance(chunk, dict):
                continue
            missing = required_metadata_keys - set(chunk.keys())
            if missing:
                logger.warning("MCP evidence chunk missing metadata fields: %s", missing)

        return results

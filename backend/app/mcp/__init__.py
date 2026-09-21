"""BIS Sahayak MCP Server Package.

Provides a Model Context Protocol (MCP) server exposing standardized read-only
tools for BIS consumer knowledge retrieval and source inspection.
"""

from app.mcp.client import BISConsumerMCPClient, MCPClientError
from app.mcp.server import mcp, run_server
from app.mcp.tools import (
    check_for_updates_impl,
    get_document_metadata_impl,
    get_source_status_impl,
    search_bis_documents_impl,
)

__all__ = [
    "mcp",
    "run_server",
    "BISConsumerMCPClient",
    "MCPClientError",
    "search_bis_documents_impl",
    "get_source_status_impl",
    "check_for_updates_impl",
    "get_document_metadata_impl",
]

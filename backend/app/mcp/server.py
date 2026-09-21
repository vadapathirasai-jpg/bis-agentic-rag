"""BIS Sahayak MCP Server.

Standalone Model Context Protocol (MCP) server providing standardized,
read-only tools for AI agents and LLMs to inspect BIS consumer knowledge,
perform semantic evidence searches, and verify document synchronization status.

Architectural Constraints:
- Pure read-only MCP interface.
- No writes to Qdrant, raw snapshots, or metadata files.
- No synthetic LLM answer generation (evidence retrieval only).
- Strict validation against the approved 9 BIS consumer sources manifest.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure backend root is on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Import MCPServer (mcp 2.x) with fallback for FastMCP (mcp 1.x)
try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

from app.mcp.tools import (
    check_for_updates_impl,
    get_document_metadata_impl,
    get_source_status_impl,
    search_bis_documents_impl,
)

logger = logging.getLogger("bis_sahayak_mcp")

# Initialize MCP server instance
mcp = MCPServer("bis-sahayak")


@mcp.tool(
    name="search_bis_documents",
    description=(
        "Search official BIS consumer standards, FAQs, portals, and hallmarking guidelines "
        "using semantic vector retrieval. Pure read-only search returning ranked evidence chunks "
        "with citation metadata (title, URL, section, score). Does NOT generate synthetic text."
    ),
)
def search_bis_documents(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Execute semantic search over the bis_consumer collection.

    Args:
        query: Natural language question or search phrase.
        top_k: Maximum number of relevant chunks to retrieve (default: 5, max: 20).
    """
    return search_bis_documents_impl(query=query, top_k=top_k)


@mcp.tool(
    name="get_source_status",
    description=(
        "Retrieve the current synchronization status, SHA-256 hashes, file sizes, "
        "and HTTP metadata for all 9 approved official BIS consumer knowledge sources."
    ),
)
def get_source_status() -> Dict[str, Any]:
    """Retrieve metadata and synchronization state of all approved BIS sources."""
    return get_source_status_impl()


@mcp.tool(
    name="check_for_updates",
    description=(
        "Perform a safe, read-only update check comparing stored metadata hashes "
        "against current snapshot/upstream state. Zero disk or Qdrant writes."
    ),
)
def check_for_updates(check_remote: bool = False) -> Dict[str, Any]:
    """Check for updates across approved BIS sources without modifying any state.

    Args:
        check_remote: If True, tests remote HTTP availability. If False (default),
            validates current local snapshot hash against stored metadata.
    """
    return check_for_updates_impl(check_remote=check_remote)


@mcp.tool(
    name="get_document_metadata",
    description=(
        "Retrieve detailed provenance and synchronization metadata for a specific "
        "approved BIS source by filename stem (e.g. 'bis_apps') or official URL. "
        "Strictly rejects unapproved or arbitrary inputs."
    ),
)
def get_document_metadata(source_id_or_url: str) -> Dict[str, Any]:
    """Retrieve metadata for a specific approved BIS source.

    Args:
        source_id_or_url: Approved source filename stem or official BIS URL.
    """
    return get_document_metadata_impl(source_id_or_url=source_id_or_url)


def run_server() -> None:
    """Run the BIS Sahayak MCP server over stdio transport."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()

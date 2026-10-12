"""
search_tool.py — Web search tool for looking up definitions and external context.

Uses DuckDuckGo search to answer questions that require external knowledge,
such as industry definitions, acronyms, or general knowledge.
"""

from typing import Dict, Any, List, Optional
import traceback


def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search the web for a query using DuckDuckGo.

    Returns
    -------
    dict with keys:
        success  : bool
        results  : list of dicts with 'title', 'body', 'href'
        query    : str
        error    : str | None
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", ""),
                })

        return {
            "success": True,
            "results": results,
            "query": query,
            "error": None,
        }

    except ImportError:
        return {
            "success": False,
            "results": [],
            "query": query,
            "error": "duckduckgo-search package not installed. Run: pip install duckduckgo-search",
        }
    except Exception as e:
        return {
            "success": False,
            "results": [],
            "query": query,
            "error": f"Search failed: {str(e)}",
        }


def format_search_results(search_result: Dict[str, Any]) -> str:
    """Format search results into a readable string for the LLM."""
    if not search_result["success"]:
        return f"Search failed: {search_result['error']}"

    if not search_result["results"]:
        return f"No results found for: {search_result['query']}"

    lines = [f"Search results for: \"{search_result['query']}\"\n"]
    for i, r in enumerate(search_result["results"], 1):
        lines.append(f"{i}. **{r['title']}**")
        lines.append(f"   {r['body']}")
        lines.append(f"   Source: {r['href']}")
        lines.append("")

    return "\n".join(lines)

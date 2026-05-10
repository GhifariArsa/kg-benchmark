"""
extractor.py — Call graph extractor for coding agent benchmark
Uses pyan3 for static analysis. Works on any Python repo.
Output: call_graph.json with nodes, edges, and metadata.
"""

import os
import sys
import json
import argparse
import glob
from pyan.analyzer import CallGraphVisitor
from pyan.node import Flavor


# Flavors we consider real, defined code entities
VALID_FLAVORS = {Flavor.FUNCTION, Flavor.METHOD, Flavor.CLASS}


def find_python_files(repo_path: str) -> list[str]:
    """Recursively find all .py files in a repo, skipping common noise dirs."""
    skip_dirs = {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
        "site-packages",
        "tests",
        "test",
        "docs",
    }

    py_files = []
    for root, dirs, files in os.walk(repo_path):
        # Prune skip dirs in-place
        dirs[:] = [
            d for d in dirs if d not in skip_dirs and not d.endswith(".egg-info")
        ]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    return sorted(py_files)


def node_id(node) -> str:
    """Canonical string ID for a node: namespace.name"""
    if node.namespace:
        return f"{node.namespace}.{node.name}"
    return node.name


def is_valid_node(node) -> bool:
    """Filter to only defined, real code entities (functions, methods, classes)."""
    return (
        node.defined
        and node.flavor in VALID_FLAVORS
        and node.namespace is not None  # exclude top-level module nodes
        and "^^^" not in node.name  # exclude internal pyan placeholders
    )


def build_call_graph(py_files: list[str], repo_path: str) -> dict:
    """
    Run pyan on the given files and return a structured call graph dict.
    """
    if not py_files:
        raise ValueError("No Python files found in repo.")

    visitor = CallGraphVisitor(py_files)

    # --- Build node registry ---
    nodes = {}
    for name, nodelist in visitor.nodes.items():
        for node in nodelist:
            if is_valid_node(node):
                nid = node_id(node)
                if nid not in nodes:
                    # Compute relative file path for readability
                    rel_file = (
                        os.path.relpath(node.filename, repo_path)
                        if node.filename
                        else None
                    )
                    nodes[nid] = {
                        "id": nid,
                        "name": node.name,
                        "namespace": node.namespace,
                        "type": node.flavor.name.lower(),  # function / method / class
                        "file": rel_file,
                    }

    # --- Build edge list ---
    edges = []
    seen_edges = set()

    for caller_node, callee_set in visitor.uses_edges.items():
        if not is_valid_node(caller_node):
            continue
        caller_id = node_id(caller_node)
        if caller_id not in nodes:
            continue

        for callee_node in callee_set:
            if not is_valid_node(callee_node):
                continue
            callee_id = node_id(callee_node)
            if callee_id not in nodes:
                continue

            edge_key = (caller_id, callee_id)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            edges.append(
                {
                    "caller": caller_id,
                    "callee": callee_id,
                }
            )

    # --- Stats ---
    stats = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "functions": sum(1 for n in nodes.values() if n["type"] == "function"),
        "methods": sum(1 for n in nodes.values() if n["type"] == "method"),
        "classes": sum(1 for n in nodes.values() if n["type"] == "class"),
        "files_analyzed": len(py_files),
    }

    return {
        "repo": os.path.basename(os.path.abspath(repo_path)),
        "stats": stats,
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract a call graph from a Python repository using pyan3."
    )
    parser.add_argument("repo_path", help="Path to the root of the Python repository.")
    parser.add_argument(
        "--output",
        "-o",
        default="call_graph.json",
        help="Output JSON file path (default: call_graph.json).",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files (default: excluded).",
    )
    args = parser.parse_args()

    repo_path = os.path.abspath(args.repo_path)
    if not os.path.isdir(repo_path):
        print(f"Error: {repo_path} is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {repo_path}")
    py_files = find_python_files(repo_path)

    if not args.include_tests:
        py_files = [
            f
            for f in py_files
            if not (
                os.path.basename(f).startswith("test_")
                or os.path.basename(f).endswith("_test.py")
            )
        ]

    print(f"Found {len(py_files)} Python files")
    if not py_files:
        print("No Python files found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("Running pyan3 static analysis...")
    try:
        graph = build_call_graph(py_files, repo_path)
    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w") as f:
        json.dump(graph, f, indent=2)

    s = graph["stats"]
    print(f"\nDone → {args.output}")
    print(
        f"  Nodes : {s['total_nodes']}  ({s['functions']} functions, {s['methods']} methods, {s['classes']} classes)"
    )
    print(f"  Edges : {s['total_edges']}")
    print(f"  Files : {s['files_analyzed']}")


if __name__ == "__main__":
    main()

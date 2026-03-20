"""
Reasoning trace module using NetworkX DAGs.

Every logical execution step is logged as a node/edge in a Directed
Acyclic Graph (DAG) and can be serialised to JSON using
``networkx.readwrite.json_graph.node_link_data``.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import networkx as nx
from networkx.readwrite import json_graph


class ReasoningTrace:
    """Directed Acyclic Graph that records the inference pipeline steps.

    Each node in the DAG carries an ``elapsed_ms`` attribute that records
    the wall-clock time (in milliseconds) from trace creation to when the
    step was added.

    Usage
    -----
    >>> trace = ReasoningTrace()
    >>> trace.add_step("root", "Ingestion", {"claims": 2})
    >>> trace.add_step("check", "PramanaCheck", {"passed": True}, parent="root")
    >>> print(trace.to_json())
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._counter: int = 0
        self._start: float = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds elapsed since this trace was created."""
        return (time.perf_counter() - self._start) * 1000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_step(
        self,
        node_id: str,
        label: str,
        data: Optional[Dict[str, Any]] = None,
        parent: Optional[str] = None,
    ) -> str:
        """Add a step node to the trace DAG.

        Parameters
        ----------
        node_id:
            Unique identifier for the node.
        label:
            Human-readable step label.
        data:
            Arbitrary metadata to store on the node.
        parent:
            If given, an edge ``parent → node_id`` is created.

        Returns
        -------
        str
            The *node_id* of the added node.
        """
        self._counter += 1
        # Prefix any data keys that would collide with reserved node attrs
        safe_data = {f"data_{k}" if k in {"label", "step_index"} else k: v
                     for k, v in (data or {}).items()}
        self._graph.add_node(
            node_id,
            label=label,
            step_index=self._counter,
            elapsed_ms=round(self.elapsed_ms, 3),
            **safe_data,
        )
        if parent is not None:
            if parent not in self._graph:
                # Auto-create missing parent as a placeholder
                self._graph.add_node(parent, label=parent, step_index=0)
            self._graph.add_edge(parent, node_id)
        return node_id

    def add_syllogism_steps(
        self,
        syllogism_trace_nodes: list,
        parent: Optional[str] = None,
    ) -> None:
        """Bulk-add nodes from a syllogism trace node list.

        Parameters
        ----------
        syllogism_trace_nodes:
            List of dicts produced by :func:`~pramana_engine.anumana.nyaya_syllogism`.
        parent:
            Optional parent node for the first step.
        """
        prev = parent
        for node_dict in syllogism_trace_nodes:
            step_num = node_dict.get("step", self._counter + 1)
            nid = f"syllogism_step_{step_num}"
            self.add_step(
                node_id=nid,
                label=node_dict.get("name", f"Step {step_num}"),
                data={"content": node_dict.get("content", "")},
                parent=prev,
            )
            prev = nid

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the DAG to a ``node_link_data`` dict.

        Returns
        -------
        Dict[str, Any]
            JSON-serialisable representation.
        """
        return json_graph.node_link_data(self._graph)

    def to_json(self, indent: int = 2) -> str:
        """Serialise the DAG to a JSON string.

        Parameters
        ----------
        indent:
            JSON indentation level.

        Returns
        -------
        str
            Pretty-printed JSON.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying :class:`networkx.DiGraph`."""
        return self._graph

    def is_dag(self) -> bool:
        """Return ``True`` if the graph is acyclic."""
        return nx.is_directed_acyclic_graph(self._graph)

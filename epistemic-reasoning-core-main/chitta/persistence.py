"""
PERSISTENCE — Serialization and Checkpointing

Provides:
- JSON export/import
- Incremental checkpointing
- Versioned snapshots
- State recovery
- Migration support

Philosophy:
- Beliefs are the source of truth
- Indexes can be rebuilt
- Provenance is preserved
- State is recoverable
"""

from __future__ import annotations

import gzip
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .belief import Belief
    from .graph import ChittaGraph
    from .utils import now_utc, to_iso
except ImportError:
    from belief import Belief
    from graph import ChittaGraph
    from utils import now_utc, to_iso


class ChittaCheckpoint:
    """
    Checkpoint manager for Chitta state.
    
    Supports:
    - Named checkpoints
    - Automatic timestamping
    - Compression
    - Metadata tracking
    """
    
    def __init__(self, checkpoint_dir: str | Path):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        graph: ChittaGraph,
        name: str | None = None,
        compress: bool = True,
        metadata: dict | None = None,
    ) -> Path:
        """
        Save checkpoint.
        
        Args:
            graph: ChittaGraph to save
            name: checkpoint name (default: timestamp)
            compress: whether to gzip compress
            metadata: additional metadata
        
        Returns:
            path to saved checkpoint
        """
        # Generate checkpoint name
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare checkpoint data
        checkpoint_data = {
            "version": "0.1.0",
            "created_at": to_iso(now_utc()),
            "name": name,
            "metadata": metadata or {},
            "graph": graph.to_dict(),
        }
        
        # Determine filename
        if compress:
            filename = f"checkpoint_{name}.json.gz"
            filepath = self.checkpoint_dir / filename
            
            # Save compressed
            with gzip.open(filepath, "wt", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        else:
            filename = f"checkpoint_{name}.json"
            filepath = self.checkpoint_dir / filename
            
            # Save uncompressed
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load(self, name: str) -> ChittaGraph:
        """
        Load checkpoint by name.
        
        Args:
            name: checkpoint name or filename
        
        Returns:
            loaded ChittaGraph
        
        Raises:
            FileNotFoundError: if checkpoint not found
        """
        # Try both compressed and uncompressed
        candidates = [
            self.checkpoint_dir / name,
            self.checkpoint_dir / f"checkpoint_{name}.json",
            self.checkpoint_dir / f"checkpoint_{name}.json.gz",
        ]
        
        filepath = None
        for candidate in candidates:
            if candidate.exists():
                filepath = candidate
                break
        
        if filepath is None:
            raise FileNotFoundError(f"Checkpoint '{name}' not found")
        
        # Load based on extension
        if filepath.suffix == ".gz":
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        
        # Reconstruct graph
        graph = ChittaGraph()
        
        # Load beliefs
        for belief_data in data["graph"]["beliefs"].values():
            belief = Belief.from_dict(belief_data)
            graph.add_belief(belief)
        
        # Load edges
        for relation, src_dict in data["graph"].get("edges", {}).items():
            for src_id, targets in src_dict.items():
                for tgt_id, weight, edge_id in targets:
                    if src_id in graph.beliefs and tgt_id in graph.beliefs:
                        graph.add_edge(src_id, relation, tgt_id, weight, edge_id)
        
        # Load metadata
        graph.metadata = data["graph"].get("metadata", {})
        
        return graph
    
    def list_checkpoints(self) -> list[dict]:
        """
        List all checkpoints.
        
        Returns:
            list of checkpoint metadata dicts
        """
        checkpoints = []
        
        for filepath in self.checkpoint_dir.glob("checkpoint_*.json*"):
            # Get file stats
            stat = filepath.stat()
            
            checkpoints.append({
                "name": filepath.stem.replace("checkpoint_", ""),
                "filename": filepath.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "compressed": filepath.suffix == ".gz",
            })
        
        # Sort by modified time (newest first)
        checkpoints.sort(key=lambda x: x["modified"], reverse=True)
        
        return checkpoints
    
    def delete(self, name: str):
        """Delete a checkpoint by name."""
        candidates = [
            self.checkpoint_dir / name,
            self.checkpoint_dir / f"checkpoint_{name}.json",
            self.checkpoint_dir / f"checkpoint_{name}.json.gz",
        ]
        
        deleted = False
        for candidate in candidates:
            if candidate.exists():
                candidate.unlink()
                deleted = True
        
        if not deleted:
            raise FileNotFoundError(f"Checkpoint '{name}' not found")
    
    def cleanup_old(self, keep_count: int = 10):
        """
        Delete old checkpoints, keeping only the most recent N.
        
        Args:
            keep_count: number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        for checkpoint in checkpoints[keep_count:]:
            filepath = self.checkpoint_dir / checkpoint["filename"]
            filepath.unlink()


class ChittaExporter:
    """
    Export Chitta to various formats.
    
    Supported formats:
    - JSON (full structure)
    - CSV (beliefs table)
    - GraphML (for visualization)
    - DOT (Graphviz)
    """
    
    @staticmethod
    def to_json(graph: ChittaGraph, filepath: str | Path, pretty: bool = True):
        """Export to JSON."""
        filepath = Path(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(graph.to_dict(), f, indent=2, ensure_ascii=False)
            else:
                json.dump(graph.to_dict(), f, ensure_ascii=False)
    
    @staticmethod
    def to_csv(graph: ChittaGraph, filepath: str | Path):
        """Export beliefs to CSV."""
        import csv
        
        filepath = Path(filepath)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "id",
                "epistemic_state",
                "template",
                "confidence",
                "statement_text",
                "entities",
                "predicates",
                "active",
                "created_at",
            ])
            
            # Rows
            for belief in graph.beliefs.values():
                writer.writerow([
                    belief.id,
                    belief.epistemic_state,
                    belief.template,
                    belief.confidence,
                    belief.statement_text,
                    "|".join(belief.entities),
                    "|".join(belief.predicates),
                    belief.active,
                    to_iso(belief.created_at),
                ])
    
    @staticmethod
    def to_graphml(graph: ChittaGraph, filepath: str | Path):
        """
        Export to GraphML format (for Gephi, Cytoscape, etc.).
        
        Note: Requires basic XML generation.
        """
        filepath = Path(filepath)
        
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
            '  <key id="confidence" for="node" attr.name="confidence" attr.type="double"/>',
            '  <key id="template" for="node" attr.name="template" attr.type="string"/>',
            '  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>',
            '  <key id="relation" for="edge" attr.name="relation" attr.type="string"/>',
            '  <graph id="chitta" edgedefault="directed">',
        ]
        
        # Add nodes
        for belief in graph.beliefs.values():
            if not belief.active:
                continue
            
            label = belief.statement_text or belief.id
            label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            
            lines.append(f'    <node id="{belief.id}">')
            lines.append(f'      <data key="label">{label}</data>')
            lines.append(f'      <data key="confidence">{belief.confidence}</data>')
            lines.append(f'      <data key="template">{belief.template}</data>')
            lines.append(f'    </node>')
        
        # Add edges
        edge_counter = 0
        for relation, src_dict in graph.edges.items():
            for src_id, targets in src_dict.items():
                if src_id not in graph.beliefs or not graph.beliefs[src_id].active:
                    continue
                
                for tgt_id, weight, _ in targets:
                    if tgt_id not in graph.beliefs or not graph.beliefs[tgt_id].active:
                        continue
                    
                    edge_id = f"e{edge_counter}"
                    edge_counter += 1
                    
                    lines.append(f'    <edge id="{edge_id}" source="{src_id}" target="{tgt_id}">')
                    lines.append(f'      <data key="relation">{relation}</data>')
                    lines.append(f'      <data key="weight">{weight}</data>')
                    lines.append(f'    </edge>')
        
        lines.append('  </graph>')
        lines.append('</graphml>')
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    @staticmethod
    def to_dot(graph: ChittaGraph, filepath: str | Path):
        """
        Export to DOT format (Graphviz).
        
        Can be rendered with: dot -Tpng output.dot -o output.png
        """
        filepath = Path(filepath)
        
        lines = [
            'digraph chitta {',
            '  rankdir=LR;',
            '  node [shape=box, style=rounded];',
        ]
        
        # Add nodes
        for belief in graph.beliefs.values():
            if not belief.active:
                continue
            
            label = belief.statement_text or belief.id
            label = label.replace('"', '\\"')
            
            color = "lightblue" if belief.epistemic_state == "asserted" else "lightyellow"
            
            lines.append(
                f'  "{belief.id}" [label="{label}\\n'
                f'conf={belief.confidence:.2f}", fillcolor="{color}", style=filled];'
            )
        
        # Add edges
        for relation, src_dict in graph.edges.items():
            for src_id, targets in src_dict.items():
                if src_id not in graph.beliefs or not graph.beliefs[src_id].active:
                    continue
                
                for tgt_id, weight, _ in targets:
                    if tgt_id not in graph.beliefs or not graph.beliefs[tgt_id].active:
                        continue
                    
                    color = "green" if relation == "supports" else "red" if relation == "contradicts" else "gray"
                    
                    lines.append(
                        f'  "{src_id}" -> "{tgt_id}" '
                        f'[label="{relation}", color="{color}"];'
                    )
        
        lines.append('}')
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


class ChittaImporter:
    """
    Import data into Chitta from various sources.
    """
    
    @staticmethod
    def from_json(filepath: str | Path) -> ChittaGraph:
        """Import from JSON file."""
        return ChittaGraph.load(filepath)
    
    @staticmethod
    def from_beliefs_list(beliefs_data: list[dict]) -> ChittaGraph:
        """
        Create graph from list of belief dictionaries.
        
        Useful for programmatic construction.
        """
        graph = ChittaGraph()
        
        for belief_data in beliefs_data:
            belief = Belief.from_dict(belief_data)
            graph.add_belief(belief)
        
        return graph

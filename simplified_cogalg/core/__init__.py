"""
Simplified CogAlg Core Algorithms

This package contains the essential algorithms for pattern discovery:
- Edge detection: Finding changes in images
- Blob detection: Grouping similar regions
- Pattern matching: Comparing patterns
- Hierarchical clustering: Building pattern hierarchies
"""

from .edge_detection import detect_edges, compute_gradient
from .blob_detection import find_blobs, Blob
from .pattern_matching import compare_patterns, Pattern, cluster_patterns
from .hierarchical_clustering import (
    build_hierarchical_graph, GraphNode, HierarchicalGraph,
    visualize_hierarchy
)

__all__ = [
    'detect_edges',
    'compute_gradient', 
    'find_blobs',
    'Blob',
    'compare_patterns',
    'Pattern',
    'cluster_patterns',
    'build_hierarchical_graph',
    'GraphNode',
    'HierarchicalGraph',
    'visualize_hierarchy'
]
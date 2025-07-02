"""
Simplified Hierarchical Clustering

This module implements recursive pattern clustering to build hierarchies.
It's a simplified version of CogAlg's agg_recursion.py.

Key concepts:
- Patterns can be grouped into higher-level patterns
- These groups can be grouped again (recursion)
- This creates a hierarchy: patterns of patterns of patterns...
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
from .pattern_matching import Pattern, compare_patterns


@dataclass
class Link:
    """
    A link represents a relationship between two nodes (patterns/graphs).
    
    Attributes:
        node1, node2: The connected nodes
        similarity: Strength of connection (0-1)
        difference: Difference between nodes
        link_type: Type of relationship ('match' or 'difference')
    """
    node1: 'GraphNode'
    node2: 'GraphNode'
    similarity: float
    difference: float
    link_type: str = 'match'
    
    def __repr__(self):
        return f"Link({self.node1.id}-{self.node2.id}, sim={self.similarity:.2f})"


@dataclass
class GraphNode:
    """
    A node in the hierarchical graph. Can contain patterns or other graphs.
    
    This is a simplified version of CogAlg's CN (Cluster Node).
    """
    id: int
    level: int = 0  # Hierarchy level (0=pattern, 1=cluster, 2=super-cluster...)
    patterns: List[Pattern] = field(default_factory=list)
    sub_nodes: List['GraphNode'] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    
    # Aggregate properties
    total_area: int = 0
    average_intensity: float = 0
    center: Tuple[float, float] = (0, 0)
    
    def add_pattern(self, pattern: Pattern):
        """Add a pattern to this node."""
        self.patterns.append(pattern)
        self._update_properties()
    
    def add_sub_node(self, node: 'GraphNode'):
        """Add a sub-node (for hierarchical structure)."""
        self.sub_nodes.append(node)
        self._update_properties()
    
    def _update_properties(self):
        """Update aggregate properties."""
        # Calculate from patterns
        if self.patterns:
            areas = [p.area for p in self.patterns]
            self.total_area = sum(areas)
            
            if self.total_area > 0:
                # Weighted average intensity
                total_intensity = sum(p.intensity * p.area for p in self.patterns)
                self.average_intensity = total_intensity / self.total_area
                
                # Weighted center
                total_y = sum(p.blobs[0].center[0] * p.area for p in self.patterns)
                total_x = sum(p.blobs[0].center[1] * p.area for p in self.patterns)
                self.center = (total_y / self.total_area, total_x / self.total_area)
        
        # Include sub-nodes
        if self.sub_nodes:
            self.total_area += sum(node.total_area for node in self.sub_nodes)
    
    def __repr__(self):
        return f"GraphNode(id={self.id}, level={self.level}, patterns={len(self.patterns)}, sub_nodes={len(self.sub_nodes)})"


@dataclass
class HierarchicalGraph:
    """
    A hierarchical graph structure for pattern organization.
    
    This represents the full hierarchy from patterns to super-clusters.
    """
    root_nodes: List[GraphNode] = field(default_factory=list)
    all_nodes: List[GraphNode] = field(default_factory=list)
    all_links: List[Link] = field(default_factory=list)
    max_level: int = 0
    
    def add_node(self, node: GraphNode):
        """Add a node to the graph."""
        if node.level == 0:
            self.root_nodes.append(node)
        self.all_nodes.append(node)
        self.max_level = max(self.max_level, node.level)
    
    def get_nodes_at_level(self, level: int) -> List[GraphNode]:
        """Get all nodes at a specific hierarchy level."""
        return [node for node in self.all_nodes if node.level == level]
    
    def __repr__(self):
        level_counts = {}
        for node in self.all_nodes:
            level_counts[node.level] = level_counts.get(node.level, 0) + 1
        return f"HierarchicalGraph(levels={self.max_level+1}, nodes={level_counts})"


def build_hierarchical_graph(patterns: List[Pattern], 
                           similarity_threshold: float = 0.6,
                           max_levels: int = 3) -> HierarchicalGraph:
    """
    Build a hierarchical graph from patterns using recursive clustering.
    
    This is a simplified version of CogAlg's recursive agglomeration.
    
    Args:
        patterns: List of base patterns
        similarity_threshold: Minimum similarity for clustering
        max_levels: Maximum hierarchy depth
    
    Returns:
        HierarchicalGraph object
    """
    graph = HierarchicalGraph()
    
    # Level 0: Convert patterns to nodes
    current_nodes = []
    for i, pattern in enumerate(patterns):
        node = GraphNode(id=i, level=0)
        node.add_pattern(pattern)
        current_nodes.append(node)
        graph.add_node(node)
    
    # Build hierarchy level by level
    for level in range(1, max_levels + 1):
        print(f"\nBuilding level {level} with {len(current_nodes)} nodes...")
        
        # Find links between nodes at current level
        links = find_node_links(current_nodes, similarity_threshold)
        graph.all_links.extend(links)
        
        # Cluster connected nodes
        clusters = cluster_nodes(current_nodes, links, level)
        
        if not clusters:
            print(f"No clusters formed at level {level}, stopping.")
            break
        
        print(f"Formed {len(clusters)} clusters at level {level}")
        
        # Add clusters to graph
        for cluster in clusters:
            graph.add_node(cluster)
        
        # Continue with next level
        current_nodes = clusters
        
        # Increase threshold for higher levels (harder to cluster)
        similarity_threshold *= 0.9
    
    return graph


def find_node_links(nodes: List[GraphNode], 
                   threshold: float) -> List[Link]:
    """
    Find links between nodes based on similarity.
    
    Args:
        nodes: List of nodes to compare
        threshold: Minimum similarity for creating a link
    
    Returns:
        List of Link objects
    """
    links = []
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            
            # Calculate similarity between nodes
            similarity = compare_nodes(node1, node2)
            
            if similarity >= threshold:
                # Create link
                difference = 1.0 - similarity
                link_type = 'match' if similarity > 0.8 else 'partial_match'
                
                link = Link(
                    node1=node1,
                    node2=node2,
                    similarity=similarity,
                    difference=difference,
                    link_type=link_type
                )
                links.append(link)
                
                # Add link references to nodes
                node1.links.append(link)
                node2.links.append(link)
    
    return links


def compare_nodes(node1: GraphNode, node2: GraphNode) -> float:
    """
    Compare two nodes and return similarity score.
    
    This is a simplified version of comp_N in CogAlg.
    
    Args:
        node1, node2: Nodes to compare
    
    Returns:
        Similarity score (0-1)
    """
    # Compare based on aggregate properties
    similarities = []
    
    # Area similarity
    if max(node1.total_area, node2.total_area) > 0:
        area_sim = min(node1.total_area, node2.total_area) / max(node1.total_area, node2.total_area)
        similarities.append(area_sim)
    
    # Intensity similarity
    if max(node1.average_intensity, node2.average_intensity) > 0:
        intensity_diff = abs(node1.average_intensity - node2.average_intensity)
        intensity_sim = 1.0 - (intensity_diff / max(node1.average_intensity, node2.average_intensity))
        similarities.append(max(0, intensity_sim))
    
    # Position similarity (based on distance)
    distance = np.sqrt((node1.center[0] - node2.center[0])**2 + 
                      (node1.center[1] - node2.center[1])**2)
    max_distance = 100  # Adjust based on image size
    position_sim = max(0, 1 - (distance / max_distance))
    similarities.append(position_sim)
    
    # Return average similarity
    return np.mean(similarities) if similarities else 0


def cluster_nodes(nodes: List[GraphNode], 
                 links: List[Link], 
                 level: int) -> List[GraphNode]:
    """
    Cluster nodes based on their links.
    
    This is a simplified version of connectivity clustering.
    
    Args:
        nodes: List of nodes to cluster
        links: Links between nodes
        level: Hierarchy level for new clusters
    
    Returns:
        List of cluster nodes
    """
    # Find connected components using links
    node_to_cluster = {}
    clusters = []
    cluster_id = 0
    
    for node in nodes:
        if node in node_to_cluster:
            continue
        
        # Start new cluster
        cluster = GraphNode(id=cluster_id + 1000 * level, level=level)
        cluster_id += 1
        
        # Flood-fill to find all connected nodes
        to_process = [node]
        processed = set()
        
        while to_process:
            current = to_process.pop()
            if current in processed:
                continue
            
            processed.add(current)
            cluster.add_sub_node(current)
            node_to_cluster[current] = cluster
            
            # Find connected nodes through links
            for link in current.links:
                other = link.node2 if link.node1 == current else link.node1
                if other in nodes and other not in processed:
                    to_process.append(other)
        
        # Only keep clusters with multiple nodes
        if len(cluster.sub_nodes) > 1:
            clusters.append(cluster)
        else:
            # Single node doesn't form a cluster
            del node_to_cluster[node]
    
    return clusters


def visualize_hierarchy(graph: HierarchicalGraph, image_shape: Tuple[int, int]):
    """
    Visualize the hierarchical graph structure.
    
    Args:
        graph: HierarchicalGraph to visualize
        image_shape: Shape of original image for scaling
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, axes = plt.subplots(1, graph.max_level + 1, 
                            figsize=(5 * (graph.max_level + 1), 5))
    
    if graph.max_level == 0:
        axes = [axes]
    
    # Colors for different hierarchy levels
    colors = plt.cm.rainbow(np.linspace(0, 1, graph.max_level + 1))
    
    for level in range(graph.max_level + 1):
        ax = axes[level]
        ax.set_xlim(0, image_shape[1])
        ax.set_ylim(image_shape[0], 0)  # Invert y-axis for image coordinates
        ax.set_aspect('equal')
        ax.set_title(f'Level {level}')
        
        # Get nodes at this level
        level_nodes = graph.get_nodes_at_level(level)
        
        for node in level_nodes:
            # Draw node
            if level == 0:
                # For patterns, draw actual blob shapes
                for pattern in node.patterns:
                    for blob in pattern.blobs:
                        y_coords = [p[0] for p in blob.pixels]
                        x_coords = [p[1] for p in blob.pixels]
                        ax.scatter(x_coords, y_coords, c=[colors[level]], 
                                 alpha=0.5, s=1)
            else:
                # For higher levels, draw bounding boxes
                all_pixels = []
                for sub_node in node.sub_nodes:
                    if sub_node.level == 0:
                        for pattern in sub_node.patterns:
                            for blob in pattern.blobs:
                                all_pixels.extend(blob.pixels)
                
                if all_pixels:
                    y_coords = [p[0] for p in all_pixels]
                    x_coords = [p[1] for p in all_pixels]
                    
                    # Draw bounding box
                    min_y, max_y = min(y_coords), max(y_coords)
                    min_x, max_x = min(x_coords), max(x_coords)
                    
                    rect = patches.Rectangle((min_x, min_y), 
                                           max_x - min_x, 
                                           max_y - min_y,
                                           linewidth=2, 
                                           edgecolor=colors[level],
                                           facecolor='none')
                    ax.add_patch(rect)
                    
                    # Add node ID
                    ax.text((min_x + max_x) / 2, min_y - 2, 
                           f'C{node.id}', 
                           ha='center', va='bottom',
                           fontsize=8, color=colors[level])
    
    plt.tight_layout()
    plt.suptitle('Hierarchical Pattern Clustering', y=1.02)
    plt.show()


# Example usage
if __name__ == "__main__":
    # This would typically use real patterns from blob detection
    print("Hierarchical Clustering Example")
    print("=" * 40)
    
    # Create some dummy patterns for demonstration
    from .blob_detection import Blob
    from .pattern_matching import Pattern
    
    # Create patterns at different locations
    patterns = []
    
    # Group 1: Similar patterns in top-left
    for i in range(3):
        blob = Blob(
            id=i,
            pixels=[(y, x) for y in range(10+i*2, 20+i*2) 
                    for x in range(10+i*2, 20+i*2)],
            intensity=50,
            area=100,
            center=(15+i*2, 15+i*2),
            bounds=(10+i*2, 10+i*2, 20+i*2, 20+i*2)
        )
        patterns.append(Pattern(id=i, blobs=[blob], features={}))
    
    # Group 2: Similar patterns in bottom-right
    for i in range(3, 6):
        blob = Blob(
            id=i,
            pixels=[(y, x) for y in range(40+i*2, 50+i*2) 
                    for x in range(40+i*2, 50+i*2)],
            intensity=150,
            area=100,
            center=(45+i*2, 45+i*2),
            bounds=(40+i*2, 40+i*2, 50+i*2, 50+i*2)
        )
        patterns.append(Pattern(id=i, blobs=[blob], features={}))
    
    # Build hierarchy
    graph = build_hierarchical_graph(patterns, similarity_threshold=0.5)
    
    print(f"\nHierarchical Graph: {graph}")
    
    # Print hierarchy structure
    for level in range(graph.max_level + 1):
        nodes = graph.get_nodes_at_level(level)
        print(f"\nLevel {level}: {len(nodes)} nodes")
        for node in nodes:
            print(f"  {node}")
            if node.sub_nodes:
                print(f"    Contains: {[n.id for n in node.sub_nodes]}")
    
    # Visualize
    visualize_hierarchy(graph, (60, 60))
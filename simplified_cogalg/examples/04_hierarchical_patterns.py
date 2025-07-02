#!/usr/bin/env python3
"""
Example 4: Hierarchical Pattern Discovery

This example demonstrates how CogAlg builds hierarchies of patterns.
It shows the recursive clustering that creates patterns of patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.blob_detection import find_blobs
from core.pattern_matching import Pattern
from core.hierarchical_clustering import (
    build_hierarchical_graph, visualize_hierarchy,
    GraphNode, HierarchicalGraph
)


def create_hierarchical_test_image():
    """
    Create an image with clear hierarchical structure:
    - Multiple similar objects that should group together
    - Groups that should form super-groups
    """
    image = np.ones((100, 150)) * 128  # Gray background
    
    # Create 4 groups of similar patterns
    
    # Group 1: Small squares (top-left quadrant)
    for i in range(3):
        for j in range(3):
            y, x = 10 + i*15, 10 + j*15
            image[y:y+8, x:x+8] = 30  # Dark squares
    
    # Group 2: Small circles approximation (top-right quadrant)
    for i in range(3):
        for j in range(3):
            y, x = 10 + i*15, 80 + j*15
            # Create circle-like pattern
            for dy in range(8):
                for dx in range(8):
                    if (dy-3.5)**2 + (dx-3.5)**2 <= 16:
                        image[y+dy, x+dx] = 30
    
    # Group 3: Large squares (bottom-left quadrant)
    for i in range(2):
        for j in range(2):
            y, x = 60 + i*20, 15 + j*20
            image[y:y+12, x:x+12] = 200  # Bright squares
    
    # Group 4: Large rectangles (bottom-right quadrant)
    for i in range(2):
        for j in range(2):
            y, x = 60 + i*20, 85 + j*20
            image[y:y+10, x:x+15] = 200  # Bright rectangles
    
    # Add some noise
    noise = np.random.normal(0, 10, image.shape)
    image = np.clip(image + noise, 0, 255)
    
    return image


def analyze_hierarchy_level(graph: HierarchicalGraph, level: int):
    """Analyze and print information about a specific hierarchy level."""
    nodes = graph.get_nodes_at_level(level)
    
    print(f"\n{'='*50}")
    print(f"Level {level} Analysis: {len(nodes)} nodes")
    print(f"{'='*50}")
    
    for node in nodes:
        print(f"\n{node}")
        
        # Show properties
        print(f"  Total area: {node.total_area} pixels")
        print(f"  Average intensity: {node.average_intensity:.1f}")
        print(f"  Center: ({node.center[0]:.1f}, {node.center[1]:.1f})")
        
        # Show composition
        if level == 0:
            print(f"  Contains {len(node.patterns)} pattern(s)")
        else:
            print(f"  Contains {len(node.sub_nodes)} sub-nodes:")
            for sub in node.sub_nodes:
                print(f"    - Node {sub.id} (level {sub.level})")
        
        # Show connections
        if node.links:
            print(f"  Connected to {len(node.links)} other nodes:")
            for link in node.links[:3]:  # Show first 3 links
                other = link.node2 if link.node1 == node else link.node1
                print(f"    - Node {other.id}: similarity={link.similarity:.2f}")


def visualize_clustering_process(image, patterns, graph):
    """Visualize the step-by-step clustering process."""
    levels = graph.max_level + 1
    fig, axes = plt.subplots(2, levels, figsize=(5*levels, 10))
    
    if levels == 1:
        axes = axes.reshape(-1, 1)
    
    # Top row: Show actual patterns/clusters
    # Bottom row: Show connectivity graphs
    
    for level in range(levels):
        # Top: Pattern visualization
        ax_top = axes[0, level]
        ax_top.imshow(image, cmap='gray', alpha=0.3)
        ax_top.set_title(f'Level {level} Patterns')
        ax_top.axis('off')
        
        # Get nodes at this level
        level_nodes = graph.get_nodes_at_level(level)
        colors = plt.cm.tab20(np.linspace(0, 1, len(level_nodes)))
        
        for node, color in zip(level_nodes, colors):
            if level == 0:
                # Draw actual patterns
                for pattern in node.patterns:
                    for blob in pattern.blobs:
                        y_coords = [p[0] for p in blob.pixels]
                        x_coords = [p[1] for p in blob.pixels]
                        ax_top.scatter(x_coords, y_coords, c=[color], 
                                     alpha=0.7, s=2)
            else:
                # Draw cluster boundaries
                all_pixels = []
                def collect_pixels(n):
                    if n.level == 0:
                        for p in n.patterns:
                            for b in p.blobs:
                                all_pixels.extend(b.pixels)
                    else:
                        for sub in n.sub_nodes:
                            collect_pixels(sub)
                
                collect_pixels(node)
                
                if all_pixels:
                    y_coords = [p[0] for p in all_pixels]
                    x_coords = [p[1] for p in all_pixels]
                    
                    # Draw convex hull or bounding box
                    from matplotlib.patches import Rectangle
                    min_y, max_y = min(y_coords), max(y_coords)
                    min_x, max_x = min(x_coords), max(x_coords)
                    
                    rect = Rectangle((min_x-1, min_y-1), 
                                   max_x-min_x+2, max_y-min_y+2,
                                   linewidth=3, edgecolor=color,
                                   facecolor=color, alpha=0.2)
                    ax_top.add_patch(rect)
                    
                    # Label
                    ax_top.text((min_x+max_x)/2, min_y-3, f'C{node.id}',
                              ha='center', color=color, fontweight='bold')
        
        # Bottom: Connectivity graph
        ax_bot = axes[1, level]
        ax_bot.set_title(f'Level {level} Connections')
        ax_bot.set_xlim(-1, image.shape[1]+1)
        ax_bot.set_ylim(image.shape[0]+1, -1)
        ax_bot.axis('off')
        
        # Draw nodes as circles at their centers
        for node, color in zip(level_nodes, colors):
            ax_bot.scatter(*node.center[::-1], s=node.total_area/5, 
                         c=[color], alpha=0.7, edgecolors='black', linewidth=2)
            ax_bot.text(node.center[1], node.center[0]-5, f'{node.id}',
                      ha='center', fontsize=8)
        
        # Draw links
        for link in graph.all_links:
            if link.node1 in level_nodes and link.node2 in level_nodes:
                y1, x1 = link.node1.center
                y2, x2 = link.node2.center
                
                # Line thickness based on similarity
                linewidth = link.similarity * 3
                alpha = link.similarity
                
                ax_bot.plot([x1, x2], [y1, y2], 'k-', 
                          linewidth=linewidth, alpha=alpha)
                
                # Show similarity value
                mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
                ax_bot.text(mid_x, mid_y, f'{link.similarity:.2f}',
                          ha='center', va='center', fontsize=6,
                          bbox=dict(boxstyle='round,pad=0.3', 
                                  facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('Hierarchical Clustering Process', y=0.98, fontsize=16)


def main():
    print("CogAlg Example 4: Hierarchical Pattern Discovery")
    print("=" * 50)
    
    # Create test image
    image = create_hierarchical_test_image()
    
    # Find initial blobs
    print("\n1. Finding initial blobs...")
    blobs = find_blobs(image, threshold=30)
    print(f"   Found {len(blobs)} blobs")
    
    # Convert to patterns
    print("\n2. Converting blobs to patterns...")
    patterns = [Pattern(id=i, blobs=[blob], features={}) 
                for i, blob in enumerate(blobs)]
    print(f"   Created {len(patterns)} patterns")
    
    # Build hierarchy
    print("\n3. Building pattern hierarchy...")
    graph = build_hierarchical_graph(
        patterns, 
        similarity_threshold=0.5,  # Lower threshold for more clustering
        max_levels=3
    )
    
    # Analyze each level
    for level in range(graph.max_level + 1):
        analyze_hierarchy_level(graph, level)
    
    # Summary statistics
    print(f"\n{'='*50}")
    print("Hierarchy Summary:")
    print(f"{'='*50}")
    print(f"Total levels: {graph.max_level + 1}")
    print(f"Total nodes: {len(graph.all_nodes)}")
    print(f"Total links: {len(graph.all_links)}")
    
    level_distribution = {}
    for node in graph.all_nodes:
        level_distribution[node.level] = level_distribution.get(node.level, 0) + 1
    
    print("\nNodes per level:")
    for level in sorted(level_distribution.keys()):
        print(f"  Level {level}: {level_distribution[level]} nodes")
    
    # Calculate clustering ratio
    if graph.max_level > 0:
        base_nodes = len(graph.get_nodes_at_level(0))
        for level in range(1, graph.max_level + 1):
            level_nodes = len(graph.get_nodes_at_level(level))
            if level_nodes > 0:
                ratio = base_nodes / level_nodes
                print(f"\nClustering ratio level 0→{level}: {ratio:.1f}:1")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # 1. Original image with all blobs
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title(f'Original Image with {len(blobs)} Detected Blobs')
    
    # Overlay blob boundaries
    for blob in blobs:
        min_y, min_x, max_y, max_x = blob.bounds
        rect = plt.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                           linewidth=1, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
    plt.axis('off')
    
    # 2. Clustering process visualization
    visualize_clustering_process(image, patterns, graph)
    
    # 3. Standard hierarchy visualization
    visualize_hierarchy(graph, image.shape)
    
    plt.show()
    
    print("\nKey Observations:")
    print("- Similar patterns are automatically grouped at level 1")
    print("- Similar groups are clustered into super-groups at level 2")
    print("- This creates a hierarchy: pixels → blobs → patterns → clusters → super-clusters")
    print("- Each level represents a higher level of abstraction")
    print("- This is how CogAlg discovers complex patterns from simple ones")


if __name__ == "__main__":
    main()
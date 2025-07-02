#!/usr/bin/env python3
"""
Simple CogAlg Demo - Text Output Version

This demonstrates the algorithm without requiring GUI display.
Results are saved to files and printed to console.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# Import our simplified modules
from core.edge_detection import detect_edges
from core.blob_detection import find_blobs, visualize_blobs
from core.pattern_matching import Pattern, compare_patterns, cluster_patterns
from core.hierarchical_clustering import build_hierarchical_graph


def create_demo_image():
    """Create a simple test image with clear patterns."""
    image = np.ones((60, 80)) * 128  # Gray background
    
    # Add some squares of different intensities
    # Dark squares (should cluster together)
    image[10:20, 10:20] = 30
    image[10:20, 30:40] = 40
    image[30:40, 10:20] = 35
    
    # Bright squares (should cluster together)
    image[30:40, 50:60] = 200
    image[45:55, 45:55] = 210
    image[45:55, 25:35] = 190
    
    return image


def save_visualization(fig, name):
    """Save figure to file."""
    filename = f"output_{name}_{datetime.now().strftime('%H%M%S')}.png"
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("SIMPLIFIED COGALG DEMONSTRATION")
    print("=" * 60)
    
    # Create test image
    print("\n1. CREATING TEST IMAGE")
    image = create_demo_image()
    print(f"  Image shape: {image.shape}")
    print(f"  Value range: {image.min():.0f} - {image.max():.0f}")
    
    # Save original image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap='gray')
    ax.set_title('Original Test Image')
    ax.axis('off')
    save_visualization(fig, 'original')
    
    # Edge detection
    print("\n2. EDGE DETECTION")
    edges = detect_edges(image, threshold=30)
    edge_count = np.sum(edges['edges'])
    print(f"  Found {edge_count} edge pixels")
    print(f"  Edge percentage: {edge_count / edges['edges'].size * 100:.1f}%")
    
    # Save edge visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(edges['gradient'], cmap='hot')
    ax1.set_title('Gradient Magnitude')
    ax1.axis('off')
    ax2.imshow(edges['edges'], cmap='binary_r')
    ax2.set_title('Detected Edges')
    ax2.axis('off')
    save_visualization(fig, 'edges')
    
    # Blob detection
    print("\n3. BLOB DETECTION")
    blobs = find_blobs(image, threshold=30)
    print(f"  Found {len(blobs)} blobs")
    
    # Print blob details
    print("\n  Blob Details:")
    print("  " + "-" * 50)
    print(f"  {'ID':>3} {'Area':>6} {'Intensity':>10} {'Center':>15}")
    print("  " + "-" * 50)
    
    for blob in sorted(blobs, key=lambda b: b.area, reverse=True)[:10]:
        center_str = f"({blob.center[0]:.0f}, {blob.center[1]:.0f})"
        print(f"  {blob.id:>3} {blob.area:>6} {blob.intensity:>10.1f} {center_str:>15}")
    
    # Save blob visualization
    blob_vis = visualize_blobs(image, blobs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(blob_vis)
    ax.set_title(f'Detected Blobs ({len(blobs)} total)')
    ax.axis('off')
    save_visualization(fig, 'blobs')
    
    # Pattern matching
    print("\n4. PATTERN CREATION AND MATCHING")
    patterns = [Pattern(id=i, blobs=[blob], features={}) 
                for i, blob in enumerate(blobs)]
    print(f"  Created {len(patterns)} patterns from blobs")
    
    # Find similar patterns
    print("\n  Finding Similar Patterns:")
    similar_pairs = []
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            match = compare_patterns(patterns[i], patterns[j])
            if match.similarity > 0.7:
                similar_pairs.append((i, j, match.similarity))
    
    print(f"  Found {len(similar_pairs)} highly similar pattern pairs (>70%)")
    for p1, p2, sim in sorted(similar_pairs, key=lambda x: x[2], reverse=True)[:5]:
        print(f"    Pattern {p1} ↔ Pattern {p2}: {sim:.1%} similarity")
    
    # Pattern clustering
    print("\n5. PATTERN CLUSTERING")
    clusters = cluster_patterns(patterns, threshold=0.7)
    print(f"  Formed {len(clusters)} clusters from {len(patterns)} patterns")
    
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:
            print(f"\n  Cluster {i}: {len(cluster)} patterns")
            areas = [p.area for p in cluster]
            intensities = [p.intensity for p in cluster]
            print(f"    Pattern IDs: {[p.id for p in cluster]}")
            print(f"    Area range: {min(areas)} - {max(areas)}")
            print(f"    Intensity range: {min(intensities):.0f} - {max(intensities):.0f}")
    
    # Hierarchical clustering
    print("\n6. HIERARCHICAL CLUSTERING")
    graph = build_hierarchical_graph(patterns, similarity_threshold=0.6, max_levels=2)
    
    print(f"\n  Hierarchy Structure:")
    for level in range(graph.max_level + 1):
        nodes = graph.get_nodes_at_level(level)
        print(f"    Level {level}: {len(nodes)} nodes")
        
        if level > 0:
            # Show what was clustered
            for node in nodes:
                sub_ids = [sub.id for sub in node.sub_nodes]
                print(f"      Node {node.id} contains: {sub_ids}")
    
    # Save hierarchy visualization
    fig, axes = plt.subplots(1, graph.max_level + 1, figsize=(5*(graph.max_level+1), 5))
    if graph.max_level == 0:
        axes = [axes]
    
    for level in range(graph.max_level + 1):
        ax = axes[level]
        ax.imshow(image, cmap='gray', alpha=0.3)
        ax.set_title(f'Hierarchy Level {level}')
        ax.axis('off')
        
        # Draw nodes at this level
        nodes = graph.get_nodes_at_level(level)
        colors = plt.cm.tab20(np.linspace(0, 1, len(nodes)))
        
        for node, color in zip(nodes, colors):
            # Collect all pixels in this node
            all_pixels = []
            
            if level == 0:
                for pattern in node.patterns:
                    for blob in pattern.blobs:
                        all_pixels.extend(blob.pixels)
            else:
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
                ax.scatter(x_coords, y_coords, c=[color], alpha=0.5, s=2)
    
    save_visualization(fig, 'hierarchy')
    
    # Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Edges detected: {edge_count}")
    print(f"  Blobs found: {len(blobs)}")
    print(f"  Patterns created: {len(patterns)}")
    print(f"  Clusters formed: {len(clusters)}")
    print(f"  Hierarchy levels: {graph.max_level + 1}")
    print(f"\n  Output files saved with prefix 'output_'")
    
    return {
        'image': image,
        'edges': edges,
        'blobs': blobs,
        'patterns': patterns,
        'clusters': clusters,
        'graph': graph
    }


if __name__ == "__main__":
    results = main()
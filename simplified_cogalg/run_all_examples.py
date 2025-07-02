#!/usr/bin/env python3
"""
Run All CogAlg Examples and Generate Report

This script runs through all the simplified CogAlg examples
and generates a comprehensive report with visualizations.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Import all our modules
from core.edge_detection import detect_edges, compute_gradient
from core.blob_detection import find_blobs, visualize_blobs, blob_statistics
from core.pattern_matching import Pattern, compare_patterns, cluster_patterns
from core.hierarchical_clustering import build_hierarchical_graph, visualize_hierarchy


def create_complex_test_image():
    """Create a more complex test image with multiple pattern types."""
    image = np.ones((120, 160)) * 128
    
    # Pattern Type 1: Small dark circles (top-left)
    for i in range(3):
        for j in range(3):
            cy, cx = 15 + i*25, 15 + j*25
            for y in range(cy-8, cy+8):
                for x in range(cx-8, cx+8):
                    if (y-cy)**2 + (x-cx)**2 <= 64:
                        image[y, x] = 40
    
    # Pattern Type 2: Medium bright squares (top-right)
    for i in range(2):
        for j in range(3):
            y, x = 10 + i*30, 95 + j*20
            image[y:y+15, x:x+15] = 200
    
    # Pattern Type 3: Large rectangles (bottom)
    for i in range(2):
        for j in range(4):
            y, x = 75 + i*25, 20 + j*35
            image[y:y+20, x:x+30] = 160 + i*20 + j*5
    
    # Add gradient region
    for i in range(20):
        image[95:105, 10+i*3:13+i*3] = 50 + i*5
    
    # Add noise
    noise = np.random.normal(0, 8, image.shape)
    image = np.clip(image + noise, 0, 255)
    
    return image


def generate_comprehensive_report():
    """Generate a comprehensive visualization report."""
    print("=" * 80)
    print("COGALG COMPREHENSIVE DEMONSTRATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = f"cogalg_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")
    
    # Create test image
    print("\n1. TEST IMAGE GENERATION")
    image = create_complex_test_image()
    print(f"   Created {image.shape} image with multiple pattern types")
    
    # Save original
    plt.figure(figsize=(10, 7))
    plt.imshow(image, cmap='gray')
    plt.title('Test Image with Multiple Pattern Types')
    plt.colorbar()
    plt.savefig(f"{output_dir}/01_original.png", dpi=150)
    plt.close()
    
    # STEP 1: Edge Detection
    print("\n2. EDGE DETECTION ANALYSIS")
    
    # Try different methods and thresholds
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    methods = ['simple', 'simple', 'simple']
    thresholds = [20, 40, 60]
    
    for idx, (method, threshold) in enumerate(zip(methods, thresholds)):
        row = idx // 3
        col = idx % 3
        
        edges = detect_edges(image, threshold=threshold)
        gradient = edges['gradient'] if method == 'simple' else compute_gradient(image, method)
        
        # Show gradient
        axes[0, col].imshow(gradient, cmap='hot')
        axes[0, col].set_title(f'Gradient (threshold={threshold})')
        axes[0, col].axis('off')
        
        # Show edges
        axes[1, col].imshow(edges['edges'], cmap='binary_r')
        axes[1, col].set_title(f'Edges (pixels={np.sum(edges["edges"])})')
        axes[1, col].axis('off')
        
        print(f"   Threshold {threshold}: {np.sum(edges['edges'])} edge pixels")
    
    plt.suptitle('Edge Detection with Different Thresholds')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_edge_comparison.png", dpi=150)
    plt.close()
    
    # STEP 2: Blob Detection
    print("\n3. BLOB DETECTION ANALYSIS")
    
    # Find blobs with optimal threshold
    blobs = find_blobs(image, threshold=40)
    print(f"   Found {len(blobs)} blobs")
    
    # Analyze blob statistics
    stats = blob_statistics(blobs)
    print(f"   Average blob area: {stats['avg_area']:.1f} pixels")
    print(f"   Blob area range: {stats['min_area']} - {stats['max_area']}")
    
    # Visualize blobs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Colored blobs
    blob_vis = visualize_blobs(image, blobs)
    ax1.imshow(blob_vis)
    ax1.set_title(f'All {len(blobs)} Detected Blobs')
    ax1.axis('off')
    
    # Blob size distribution
    areas = [blob.area for blob in blobs]
    ax2.hist(areas, bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Blob Area (pixels)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Blob Sizes')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_blob_analysis.png", dpi=150)
    plt.close()
    
    # STEP 3: Pattern Analysis
    print("\n4. PATTERN MATCHING ANALYSIS")
    
    # Create patterns
    patterns = [Pattern(id=i, blobs=[blob], features={}) for i, blob in enumerate(blobs)]
    
    # Compute similarity matrix
    n = min(len(patterns), 50)  # Limit for visualization
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                match = compare_patterns(patterns[i], patterns[j])
                similarity_matrix[i, j] = match.similarity
            else:
                similarity_matrix[i, j] = 1.0
    
    # Visualize similarity matrix
    plt.figure(figsize=(10, 8))
    im = plt.imshow(similarity_matrix, cmap='hot', aspect='auto')
    plt.colorbar(im, label='Similarity')
    plt.title('Pattern Similarity Matrix')
    plt.xlabel('Pattern ID')
    plt.ylabel('Pattern ID')
    plt.savefig(f"{output_dir}/04_similarity_matrix.png", dpi=150)
    plt.close()
    
    # Find and report highly similar patterns
    high_similarity_count = np.sum(similarity_matrix > 0.8) - n  # Subtract diagonal
    print(f"   Highly similar pattern pairs (>80%): {high_similarity_count//2}")
    
    # STEP 4: Pattern Clustering
    print("\n5. PATTERN CLUSTERING ANALYSIS")
    
    clusters = cluster_patterns(patterns, threshold=0.7)
    print(f"   Formed {len(clusters)} clusters from {len(patterns)} patterns")
    
    # Visualize clusters
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Cluster visualization on image
    cluster_img = np.zeros((*image.shape, 3))
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
    
    for cluster_id, (cluster, color) in enumerate(zip(clusters, colors)):
        for pattern in cluster:
            for blob in pattern.blobs:
                for y, x in blob.pixels:
                    cluster_img[y, x] = color[:3]
    
    ax1.imshow(image, cmap='gray', alpha=0.5)
    ax1.imshow(cluster_img, alpha=0.7)
    ax1.set_title(f'Pattern Clusters ({len(clusters)} clusters)')
    ax1.axis('off')
    
    # Cluster size distribution
    cluster_sizes = [len(cluster) for cluster in clusters]
    ax2.bar(range(len(clusters)), cluster_sizes)
    ax2.set_xlabel('Cluster ID')
    ax2.set_ylabel('Number of Patterns')
    ax2.set_title('Patterns per Cluster')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_clustering.png", dpi=150)
    plt.close()
    
    # STEP 5: Hierarchical Analysis
    print("\n6. HIERARCHICAL CLUSTERING ANALYSIS")
    
    graph = build_hierarchical_graph(patterns, similarity_threshold=0.6, max_levels=3)
    
    print(f"   Built hierarchy with {graph.max_level + 1} levels")
    for level in range(graph.max_level + 1):
        nodes = graph.get_nodes_at_level(level)
        print(f"   Level {level}: {len(nodes)} nodes")
    
    # Custom hierarchy visualization
    fig, axes = plt.subplots(1, graph.max_level + 1, figsize=(5*(graph.max_level+1), 5))
    if graph.max_level == 0:
        axes = [axes]
    
    for level in range(graph.max_level + 1):
        ax = axes[level]
        ax.imshow(image, cmap='gray', alpha=0.3)
        ax.set_title(f'Level {level} ({len(graph.get_nodes_at_level(level))} nodes)')
        ax.axis('off')
        
        # Draw nodes at this level
        nodes = graph.get_nodes_at_level(level)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(nodes)))
        
        for node, color in zip(nodes, colors):
            # Collect all pixels
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
                
                # Draw bounding box
                min_y, max_y = min(y_coords), max(y_coords)
                min_x, max_x = min(x_coords), max(x_coords)
                
                from matplotlib.patches import Rectangle
                rect = Rectangle((min_x-1, min_y-1), 
                               max_x-min_x+2, max_y-min_y+2,
                               linewidth=2, edgecolor=color,
                               facecolor=color, alpha=0.2)
                ax.add_patch(rect)
                
                # Add label
                ax.text((min_x+max_x)/2, min_y-3, f'{node.id}',
                       ha='center', color=color, fontweight='bold', fontsize=10)
    
    plt.suptitle('Hierarchical Pattern Organization')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_hierarchy.png", dpi=150)
    plt.close()
    
    # Generate summary report
    print("\n7. GENERATING SUMMARY REPORT")
    
    with open(f"{output_dir}/summary_report.txt", 'w') as f:
        f.write("COGALG SIMPLIFIED ALGORITHM - ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("IMAGE STATISTICS:\n")
        f.write(f"  Dimensions: {image.shape}\n")
        f.write(f"  Value range: {image.min():.0f} - {image.max():.0f}\n")
        f.write(f"  Mean intensity: {image.mean():.1f}\n\n")
        
        f.write("PROCESSING RESULTS:\n")
        f.write(f"  Edge pixels detected: {np.sum(detect_edges(image, 40)['edges'])}\n")
        f.write(f"  Blobs found: {len(blobs)}\n")
        f.write(f"  Patterns created: {len(patterns)}\n")
        f.write(f"  Clusters formed: {len(clusters)}\n")
        f.write(f"  Hierarchy levels: {graph.max_level + 1}\n\n")
        
        f.write("BLOB STATISTICS:\n")
        for key, value in stats.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nCLUSTER DETAILS:\n")
        for i, cluster in enumerate(clusters):
            f.write(f"  Cluster {i}: {len(cluster)} patterns\n")
            if len(cluster) > 1:
                areas = [p.area for p in cluster]
                f.write(f"    Area range: {min(areas)} - {max(areas)}\n")
        
        f.write("\nHIERARCHY STRUCTURE:\n")
        for level in range(graph.max_level + 1):
            nodes = graph.get_nodes_at_level(level)
            f.write(f"  Level {level}: {len(nodes)} nodes\n")
    
    print(f"\n   Summary report saved to: {output_dir}/summary_report.txt")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - 01_original.png: Test image")
    print("  - 02_edge_comparison.png: Edge detection analysis")
    print("  - 03_blob_analysis.png: Blob detection results")
    print("  - 04_similarity_matrix.png: Pattern similarity visualization")
    print("  - 05_clustering.png: Pattern clustering results")
    print("  - 06_hierarchy.png: Hierarchical organization")
    print("  - summary_report.txt: Detailed analysis report")


if __name__ == "__main__":
    generate_comprehensive_report()
#!/usr/bin/env python3
"""
Example 3: Pattern Matching

This example demonstrates how to compare patterns (blobs) to find
similarities. This is how CogAlg recognizes recurring structures.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.blob_detection import find_blobs, visualize_blobs
from core.pattern_matching import Pattern, compare_patterns, find_similar_patterns, cluster_patterns


def create_test_image_with_patterns():
    """Create an image with repeated patterns."""
    image = np.ones((50, 80)) * 128  # Gray background
    
    # Create similar squares (pattern type 1)
    square_value = 50
    image[5:10, 5:10] = square_value      # Square 1
    image[5:10, 20:25] = square_value     # Square 2 (same size)
    image[25:30, 15:20] = square_value    # Square 3 (same size)
    image[35:40, 35:40] = square_value    # Square 4 (same size)
    
    # Create similar rectangles (pattern type 2)
    rect_value = 200
    image[15:18, 40:48] = rect_value      # Rectangle 1
    image[30:33, 50:58] = rect_value      # Rectangle 2 (same shape)
    image[40:43, 5:13] = rect_value       # Rectangle 3 (same shape)
    
    # Create unique shapes
    image[20:23, 65:75] = 20              # Wide dark rectangle
    image[40:48, 65:68] = 250             # Tall bright rectangle
    
    # Add some noise
    noise = np.random.normal(0, 5, image.shape)
    image = np.clip(image + noise, 0, 255)
    
    return image


def visualize_pattern_matches(image, patterns, matches):
    """Visualize pattern matching results."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Show original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Show all patterns
    pattern_img = np.zeros((*image.shape, 3))
    colors = plt.cm.tab20(np.linspace(0, 1, len(patterns)))
    
    for pattern, color in zip(patterns, colors):
        for blob in pattern.blobs:
            for y, x in blob.pixels:
                pattern_img[y, x] = color[:3]
    
    axes[0, 1].imshow(pattern_img)
    axes[0, 1].set_title(f'All {len(patterns)} Patterns')
    axes[0, 1].axis('off')
    
    # Show similarity matrix
    n = len(patterns)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                match = compare_patterns(patterns[i], patterns[j])
                similarity_matrix[i, j] = match.similarity
            else:
                similarity_matrix[i, j] = 1.0
    
    im = axes[0, 2].imshow(similarity_matrix, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Pattern Similarity Matrix')
    axes[0, 2].set_xlabel('Pattern ID')
    axes[0, 2].set_ylabel('Pattern ID')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Show top matches
    if matches:
        # Highlight query pattern
        query_img = image.copy()
        for blob in matches[0].pattern1.blobs:
            for y, x in blob.pixels:
                query_img[y, x] = 255
        
        axes[1, 0].imshow(query_img, cmap='gray')
        axes[1, 0].set_title('Query Pattern (white)')
        axes[1, 0].axis('off')
        
        # Show best matches
        for idx, match in enumerate(matches[:2]):
            match_img = image.copy()
            # Highlight matched pattern
            for blob in match.pattern2.blobs:
                for y, x in blob.pixels:
                    match_img[y, x] = 255
            
            axes[1, idx+1].imshow(match_img, cmap='gray')
            axes[1, idx+1].set_title(f'Match {idx+1}: {match.similarity:.2%}\n({match.match_type})')
            axes[1, idx+1].axis('off')
    
    plt.tight_layout()
    return similarity_matrix


def main():
    print("CogAlg Example 3: Pattern Matching")
    print("=" * 40)
    
    # Create test image
    image = create_test_image_with_patterns()
    
    # Find blobs
    blobs = find_blobs(image, threshold=30)
    print(f"Found {len(blobs)} blobs")
    
    # Convert blobs to patterns
    patterns = [Pattern(id=i, blobs=[blob], features={}) 
                for i, blob in enumerate(blobs)]
    
    # Analyze pattern properties
    print("\nPattern Properties:")
    print("-" * 40)
    print(f"{'ID':>3} {'Area':>6} {'Intensity':>10} {'Center':>15}")
    print("-" * 40)
    
    for p in patterns:
        center = f"({p.blobs[0].center[0]:.0f}, {p.blobs[0].center[1]:.0f})"
        print(f"{p.id:>3} {p.area:>6} {p.intensity:>10.1f} {center:>15}")
    
    # Compare all patterns
    print("\nPattern Comparisons:")
    print("-" * 40)
    
    match_count = 0
    high_similarity_pairs = []
    
    for i in range(len(patterns)):
        for j in range(i+1, len(patterns)):
            match = compare_patterns(patterns[i], patterns[j])
            
            if match.similarity > 0.7:  # High similarity threshold
                match_count += 1
                high_similarity_pairs.append((i, j, match))
                print(f"Pattern {i} <-> Pattern {j}:")
                print(f"  Similarity: {match.similarity:.2%}")
                print(f"  Match type: {match.match_type}")
                print(f"  Details: {match.details}")
    
    print(f"\nFound {match_count} high-similarity pairs (>70%)")
    
    # Find patterns similar to first square
    print("\n" + "="*40)
    print("Finding patterns similar to Pattern 0:")
    
    similar = find_similar_patterns(patterns[0], patterns[1:], threshold=0.6)
    print(f"Found {len(similar)} similar patterns")
    
    for i, match in enumerate(similar):
        print(f"\n{i+1}. Pattern {match.pattern2.id}:")
        print(f"   Similarity: {match.similarity:.2%}")
        print(f"   Best matching aspect: {match.match_type}")
    
    # Cluster patterns
    print("\n" + "="*40)
    print("Clustering Similar Patterns:")
    
    clusters = cluster_patterns(patterns, threshold=0.7)
    print(f"Formed {len(clusters)} clusters")
    
    for i, cluster in enumerate(clusters):
        areas = [p.area for p in cluster]
        intensities = [p.intensity for p in cluster]
        
        print(f"\nCluster {i}: {len(cluster)} patterns")
        print(f"  Pattern IDs: {[p.id for p in cluster]}")
        print(f"  Average area: {np.mean(areas):.1f}")
        print(f"  Average intensity: {np.mean(intensities):.1f}")
        
        if len(cluster) > 1:
            print(f"  → This cluster contains similar patterns!")
    
    # Visualize results
    similarity_matrix = visualize_pattern_matches(image, patterns, similar)
    
    # Additional visualization: Pattern clusters
    fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    cluster_img = np.zeros((*image.shape, 3))
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
    
    for cluster_id, (cluster, color) in enumerate(zip(clusters, cluster_colors)):
        for pattern in cluster:
            for blob in pattern.blobs:
                for y, x in blob.pixels:
                    cluster_img[y, x] = color[:3]
    
    ax.imshow(cluster_img)
    ax.set_title(f'Pattern Clusters ({len(clusters)} clusters)\nSame color = similar patterns')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nKey Observations:")
    print("- Similar shapes are automatically grouped together")
    print("- Matching considers multiple aspects (shape, intensity, position)")
    print("- Clustering creates higher-level pattern groups")
    print("- This is how CogAlg builds pattern hierarchies")


if __name__ == "__main__":
    main()
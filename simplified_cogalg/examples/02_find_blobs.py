#!/usr/bin/env python3
"""
Example 2: Blob Detection

This example shows how to find and analyze blobs (connected regions)
in an image. Blobs are the basic building blocks for pattern discovery.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.blob_detection import find_blobs, visualize_blobs, blob_statistics
from core.edge_detection import compute_gradient


def create_test_image():
    """Create a test image with multiple distinct regions."""
    image = np.ones((30, 40)) * 50  # Gray background
    
    # Add dark regions
    image[5:10, 5:15] = 10    # Dark rectangle
    image[8:13, 25:30] = 20   # Another dark region
    
    # Add bright regions
    image[18:25, 8:15] = 200  # Bright rectangle
    image[20:24, 20:28] = 180 # Another bright region
    
    # Add a gradient region
    for i in range(10):
        image[15:17, 30+i] = 50 + i * 10
    
    return image


def analyze_blob(blob, image):
    """Print detailed information about a blob."""
    print(f"\nBlob {blob.id}:")
    print(f"  Area: {blob.area} pixels")
    print(f"  Intensity: {blob.intensity:.1f}")
    print(f"  Center: ({blob.center[0]:.1f}, {blob.center[1]:.1f})")
    print(f"  Bounding box: {blob.bounds}")
    
    # Calculate additional properties
    min_y, min_x, max_y, max_x = blob.bounds
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    aspect_ratio = width / height
    
    print(f"  Dimensions: {width}x{height}")
    print(f"  Aspect ratio: {aspect_ratio:.2f}")
    
    # Calculate intensity variance within blob
    intensities = [image[y, x] for y, x in blob.pixels]
    variance = np.var(intensities)
    print(f"  Intensity variance: {variance:.2f}")


def main():
    print("CogAlg Example 2: Blob Detection")
    print("=" * 40)
    
    # Create test image
    image = create_test_image()
    
    # Find blobs with different thresholds
    thresholds = [10, 30, 50]
    
    fig, axes = plt.subplots(2, len(thresholds) + 1, figsize=(14, 8))
    
    # Show original image and gradient
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    gradient = compute_gradient(image)
    axes[1, 0].imshow(gradient, cmap='hot')
    axes[1, 0].set_title('Gradient Map')
    axes[1, 0].axis('off')
    
    # Try different thresholds
    all_stats = []
    
    for i, threshold in enumerate(thresholds):
        blobs = find_blobs(image, threshold=threshold)
        
        # Visualize blobs
        blob_vis = visualize_blobs(image, blobs)
        axes[0, i+1].imshow(blob_vis)
        axes[0, i+1].set_title(f'Blobs (threshold={threshold})\nCount: {len(blobs)}')
        axes[0, i+1].axis('off')
        
        # Show blob boundaries on original
        boundary_img = image.copy()
        for blob in blobs:
            # Draw bounding boxes
            min_y, min_x, max_y, max_x = blob.bounds
            boundary_img[min_y:max_y+1, min_x] = 255
            boundary_img[min_y:max_y+1, max_x] = 255
            boundary_img[min_y, min_x:max_x+1] = 255
            boundary_img[max_y, min_x:max_x+1] = 255
        
        axes[1, i+1].imshow(boundary_img, cmap='gray')
        axes[1, i+1].set_title('Blob Boundaries')
        axes[1, i+1].axis('off')
        
        # Calculate statistics
        stats = blob_statistics(blobs)
        all_stats.append((threshold, stats))
        
        print(f"\nThreshold {threshold}:")
        print(f"  Found {stats.get('count', 0)} blobs")
        if stats:
            print(f"  Average area: {stats['avg_area']:.1f} pixels")
            print(f"  Area range: {stats['min_area']} - {stats['max_area']}")
            print(f"  Intensity range: {stats['intensity_range'][0]:.1f} - {stats['intensity_range'][1]:.1f}")
    
    plt.tight_layout()
    
    # Detailed analysis of blobs from middle threshold
    print("\n" + "="*40)
    print("Detailed Blob Analysis (threshold=30):")
    blobs = find_blobs(image, threshold=30)
    
    # Sort blobs by area
    blobs_sorted = sorted(blobs, key=lambda b: b.area, reverse=True)
    
    # Analyze top 3 largest blobs
    for blob in blobs_sorted[:3]:
        analyze_blob(blob, image)
    
    # Show blob size distribution
    fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
    areas = [blob.area for blob in blobs]
    ax.hist(areas, bins=20, edgecolor='black')
    ax.set_xlabel('Blob Area (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Blob Sizes')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nKey Observations:")
    print("- Lower thresholds create fewer, larger blobs (more merging)")
    print("- Higher thresholds create more, smaller blobs (less merging)")
    print("- Blob properties help identify different image regions")
    print("- This segmentation is the foundation for pattern matching")


if __name__ == "__main__":
    main()
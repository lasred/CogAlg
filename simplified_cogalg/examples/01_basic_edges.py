#!/usr/bin/env python3
"""
Example 1: Basic Edge Detection

This example demonstrates the simplest operation in CogAlg:
finding edges (boundaries) in an image.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.edge_detection import detect_edges, compute_gradient


def create_test_image():
    """Create a simple test image with clear edges."""
    image = np.zeros((20, 20))
    
    # Add a square
    image[5:10, 5:10] = 100
    
    # Add a rectangle  
    image[12:18, 3:8] = 150
    
    # Add a diagonal line
    for i in range(8):
        image[5+i, 12+i] = 200
        
    return image


def main():
    print("CogAlg Example 1: Basic Edge Detection")
    print("=" * 40)
    
    # Create test image
    image = create_test_image()
    
    # Detect edges with different thresholds
    thresholds = [30, 50, 80]
    
    # Create visualization
    fig, axes = plt.subplots(2, len(thresholds) + 1, figsize=(12, 6))
    
    # Show original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Show gradient
    gradient = compute_gradient(image)
    axes[1, 0].imshow(gradient, cmap='hot')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].axis('off')
    
    # Show edge detection with different thresholds
    for i, threshold in enumerate(thresholds):
        result = detect_edges(image, threshold=threshold)
        
        # Show gradient
        axes[0, i+1].imshow(result['gradient'], cmap='hot')
        axes[0, i+1].set_title(f'Gradient\n(threshold={threshold})')
        axes[0, i+1].axis('off')
        
        # Show edges
        axes[1, i+1].imshow(result['edges'], cmap='binary_r')
        axes[1, i+1].set_title(f'Edges\n(threshold={threshold})')
        axes[1, i+1].axis('off')
        
        # Print statistics
        edge_pixels = np.sum(result['edges'])
        print(f"\nThreshold {threshold}:")
        print(f"  Edge pixels: {edge_pixels}")
        print(f"  Edge percentage: {edge_pixels / result['edges'].size * 100:.1f}%")
    
    plt.tight_layout()
    plt.suptitle('Edge Detection with Different Thresholds', y=1.02)
    
    # Show directional gradients
    fig2, axes2 = plt.subplots(1, 3, figsize=(10, 4))
    
    result = detect_edges(image, threshold=50)
    
    axes2[0].imshow(image, cmap='gray')
    axes2[0].set_title('Original')
    axes2[0].axis('off')
    
    axes2[1].imshow(result['dx'], cmap='RdBu_r')
    axes2[1].set_title('Horizontal Gradient (dx)')
    axes2[1].axis('off')
    
    axes2[2].imshow(result['dy'], cmap='RdBu_r')
    axes2[2].set_title('Vertical Gradient (dy)')
    axes2[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\nKey Observations:")
    print("- Lower thresholds detect more edges (more sensitive)")
    print("- Higher thresholds only detect strong edges")
    print("- Gradients show the direction of change")
    print("- This is the foundation for all higher-level processing")


if __name__ == "__main__":
    main()
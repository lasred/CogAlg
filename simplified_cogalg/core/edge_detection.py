"""
Simple Edge Detection

This module finds edges (boundaries) in images by comparing neighboring pixels.
An edge is where the image intensity changes significantly.
"""

import numpy as np


def detect_edges(image, threshold=10):
    """
    Detect edges in an image using simple gradient calculation.
    
    Args:
        image: 2D numpy array representing grayscale image
        threshold: Minimum change to consider as an edge
    
    Returns:
        Dictionary with:
        - 'gradient': 2D array of edge strengths
        - 'edges': Binary array (True where edges exist)
        - 'dx': Horizontal changes
        - 'dy': Vertical changes
    """
    # Ensure image is float for calculations
    image = image.astype(float)
    height, width = image.shape
    
    # Initialize arrays for gradients
    dx = np.zeros((height-1, width-1))  # Horizontal gradient
    dy = np.zeros((height-1, width-1))  # Vertical gradient
    
    # Calculate gradients by comparing neighbors
    for y in range(height-1):
        for x in range(width-1):
            # Horizontal change (left to right)
            dx[y, x] = image[y, x+1] - image[y, x]
            
            # Vertical change (top to bottom)
            dy[y, x] = image[y+1, x] - image[y, x]
    
    # Calculate total gradient (edge strength)
    gradient = np.sqrt(dx**2 + dy**2)
    
    # Find edges (where gradient exceeds threshold)
    edges = gradient > threshold
    
    return {
        'gradient': gradient,
        'edges': edges,
        'dx': dx,
        'dy': dy
    }


def compute_gradient(image, method='simple'):
    """
    Compute image gradient using different methods.
    
    Args:
        image: 2D numpy array
        method: 'simple' or 'sobel'
    
    Returns:
        Gradient magnitude array
    """
    if method == 'simple':
        # Simple 2x2 kernel gradient (like original CogAlg)
        height, width = image.shape
        gradient = np.zeros((height-1, width-1))
        
        for y in range(height-1):
            for x in range(width-1):
                # Compare diagonal pixels
                diff1 = abs(image[y+1, x+1] - image[y, x])
                diff2 = abs(image[y+1, x] - image[y, x+1])
                gradient[y, x] = (diff1 + diff2) / 2
                
        return gradient
    
    elif method == 'sobel':
        # Sobel edge detection (more sophisticated)
        from scipy import ndimage
        dx = ndimage.sobel(image, axis=1)
        dy = ndimage.sobel(image, axis=0)
        return np.sqrt(dx**2 + dy**2)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_edges(image, edges):
    """
    Create visualization of edges overlaid on original image.
    
    Args:
        image: Original image
        edges: Binary edge array
    
    Returns:
        RGB image with edges highlighted
    """
    import matplotlib.pyplot as plt
    
    # Create RGB version of image
    rgb = np.stack([image] * 3, axis=-1)
    
    # Highlight edges in red
    if edges.shape != image.shape:
        # Pad edges to match image size if needed
        padded_edges = np.zeros_like(image, dtype=bool)
        padded_edges[:edges.shape[0], :edges.shape[1]] = edges
        edges = padded_edges
    
    rgb[edges] = [255, 0, 0]  # Red for edges
    
    return rgb


# Example usage
if __name__ == "__main__":
    # Create a simple test image with an edge
    test_image = np.array([
        [10, 10, 10, 50, 50],
        [10, 10, 10, 50, 50],
        [10, 10, 10, 50, 50],
        [20, 20, 20, 60, 60],
        [20, 20, 20, 60, 60]
    ])
    
    # Detect edges
    result = detect_edges(test_image, threshold=20)
    
    print("Gradient map:")
    print(result['gradient'])
    print("\nEdges found at:")
    print(result['edges'])
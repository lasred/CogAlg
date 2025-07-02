"""
Simple Blob Detection

This module groups pixels into "blobs" - connected regions with similar properties.
Like finding clouds in the sky or islands in the ocean.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set


@dataclass
class Blob:
    """
    A blob is a connected region of similar pixels.
    
    Attributes:
        id: Unique identifier
        pixels: List of (y, x) coordinates
        intensity: Average pixel value
        area: Number of pixels
        center: (y, x) center of mass
        bounds: (min_y, min_x, max_y, max_x) bounding box
    """
    id: int
    pixels: List[Tuple[int, int]]
    intensity: float
    area: int
    center: Tuple[float, float]
    bounds: Tuple[int, int, int, int]
    
    def __repr__(self):
        return f"Blob(id={self.id}, area={self.area}, intensity={self.intensity:.1f})"


def find_blobs(image, gradient=None, threshold=10):
    """
    Find blobs (connected regions) in an image.
    
    This is a simplified version of CogAlg's flood-fill algorithm.
    
    Args:
        image: 2D numpy array
        gradient: Optional pre-computed gradient
        threshold: Gradient threshold for blob boundaries
    
    Returns:
        List of Blob objects
    """
    height, width = image.shape
    
    # Compute gradient if not provided
    if gradient is None:
        from .edge_detection import compute_gradient
        gradient = compute_gradient(image)
    
    # Pad gradient to match image size if needed
    if gradient.shape != image.shape:
        padded_gradient = np.zeros_like(image)
        padded_gradient[:gradient.shape[0], :gradient.shape[1]] = gradient
        gradient = padded_gradient
    
    # Create mask for visited pixels
    visited = np.zeros((height, width), dtype=bool)
    
    # Find blobs
    blobs = []
    blob_id = 0
    
    for y in range(height):
        for x in range(width):
            if not visited[y, x]:
                # Start new blob from this pixel
                blob_pixels = _flood_fill(image, gradient, visited, y, x, threshold)
                
                if len(blob_pixels) > 0:
                    # Create blob object
                    blob = _create_blob(blob_id, blob_pixels, image)
                    blobs.append(blob)
                    blob_id += 1
    
    return blobs


def _flood_fill(image, gradient, visited, start_y, start_x, threshold):
    """
    Flood fill algorithm to find connected pixels.
    
    Args:
        image: Original image
        gradient: Gradient array
        visited: Boolean array of visited pixels
        start_y, start_x: Starting position
        threshold: Gradient threshold
    
    Returns:
        List of (y, x) coordinates in the blob
    """
    height, width = image.shape
    pixels = []
    stack = [(start_y, start_x)]
    
    # Determine if this is a "flat" blob (low gradient) or "edge" blob
    is_flat = gradient[start_y, start_x] <= threshold
    
    while stack:
        y, x = stack.pop()
        
        # Check bounds and if already visited
        if (y < 0 or y >= height or x < 0 or x >= width or visited[y, x]):
            continue
            
        # Check if pixel belongs to same type of blob
        pixel_is_flat = gradient[y, x] <= threshold
        if pixel_is_flat != is_flat:
            continue
            
        # Add pixel to blob
        visited[y, x] = True
        pixels.append((y, x))
        
        # Check 4-connected neighbors (not diagonal for simplicity)
        neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
        
        # For flat blobs, also check diagonals (8-connected)
        if is_flat:
            neighbors.extend([(y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)])
        
        stack.extend(neighbors)
    
    return pixels


def _create_blob(blob_id, pixels, image):
    """
    Create a Blob object from a list of pixels.
    
    Args:
        blob_id: Unique identifier
        pixels: List of (y, x) coordinates
        image: Original image for intensity values
    
    Returns:
        Blob object
    """
    # Calculate blob properties
    intensities = [image[y, x] for y, x in pixels]
    avg_intensity = np.mean(intensities)
    
    # Calculate center of mass
    y_coords = [p[0] for p in pixels]
    x_coords = [p[1] for p in pixels]
    center_y = np.mean(y_coords)
    center_x = np.mean(x_coords)
    
    # Calculate bounding box
    min_y, max_y = min(y_coords), max(y_coords)
    min_x, max_x = min(x_coords), max(x_coords)
    
    return Blob(
        id=blob_id,
        pixels=pixels,
        intensity=avg_intensity,
        area=len(pixels),
        center=(center_y, center_x),
        bounds=(min_y, min_x, max_y, max_x)
    )


def visualize_blobs(image, blobs):
    """
    Visualize blobs with different colors.
    
    Args:
        image: Original image
        blobs: List of Blob objects
    
    Returns:
        RGB image with colored blobs
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Create output image
    output = np.zeros((*image.shape, 3))
    
    # Color each blob differently
    colors = cm.rainbow(np.linspace(0, 1, len(blobs)))
    
    for blob, color in zip(blobs, colors):
        for y, x in blob.pixels:
            output[y, x] = color[:3] * 255
    
    return output.astype(np.uint8)


def blob_statistics(blobs):
    """
    Calculate statistics about the blobs.
    
    Args:
        blobs: List of Blob objects
    
    Returns:
        Dictionary of statistics
    """
    if not blobs:
        return {}
    
    areas = [b.area for b in blobs]
    intensities = [b.intensity for b in blobs]
    
    return {
        'count': len(blobs),
        'total_area': sum(areas),
        'avg_area': np.mean(areas),
        'min_area': min(areas),
        'max_area': max(areas),
        'avg_intensity': np.mean(intensities),
        'intensity_range': (min(intensities), max(intensities))
    }


# Example usage
if __name__ == "__main__":
    # Create test image with distinct regions
    test_image = np.array([
        [10, 10, 10, 50, 50],
        [10, 10, 10, 50, 50],
        [10, 10, 10, 50, 50],
        [70, 70, 30, 30, 30],
        [70, 70, 30, 30, 30]
    ])
    
    # Find blobs
    blobs = find_blobs(test_image, threshold=20)
    
    print(f"Found {len(blobs)} blobs:")
    for blob in blobs:
        print(f"  {blob}")
    
    # Print statistics
    stats = blob_statistics(blobs)
    print(f"\nStatistics: {stats}")
"""
Simple CogAlg - Main Interface

This module provides a simple, high-level interface to the CogAlg
pattern discovery algorithm.
"""

import numpy as np
from typing import List, Optional, Union
import matplotlib.pyplot as plt

from core.edge_detection import detect_edges, compute_gradient
from core.blob_detection import find_blobs, visualize_blobs, Blob
from core.pattern_matching import Pattern, compare_patterns, cluster_patterns


class CogAlgResult:
    """Container for CogAlg processing results."""
    
    def __init__(self, image, edges, blobs, patterns, clusters):
        self.image = image
        self.edges = edges
        self.blobs = blobs
        self.patterns = patterns
        self.clusters = clusters
    
    def show_blobs(self, title="Detected Blobs"):
        """Display the detected blobs."""
        blob_img = visualize_blobs(self.image, self.blobs)
        plt.figure(figsize=(10, 8))
        plt.imshow(blob_img)
        plt.title(f"{title} (Count: {len(self.blobs)})")
        plt.axis('off')
        plt.show()
    
    def show_edges(self, title="Edge Detection"):
        """Display the edge detection results."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(self.edges['gradient'], cmap='hot')
        plt.title("Gradient")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(self.edges['edges'], cmap='binary_r')
        plt.title("Edges")
        plt.axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def show_clusters(self, title="Pattern Clusters"):
        """Display the pattern clusters."""
        # Create cluster visualization
        cluster_img = np.zeros((*self.image.shape, 3))
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.clusters)))
        
        for cluster_id, (cluster, color) in enumerate(zip(self.clusters, colors)):
            for pattern in cluster:
                for blob in pattern.blobs:
                    for y, x in blob.pixels:
                        cluster_img[y, x] = color[:3]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cluster_img)
        plt.title(f"{title} (Count: {len(self.clusters)})")
        plt.axis('off')
        plt.show()
    
    def show_all(self):
        """Display all processing stages."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original
        axes[0, 0].imshow(self.image, cmap='gray')
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Edges
        axes[0, 1].imshow(self.edges['gradient'], cmap='hot')
        axes[0, 1].set_title("Edge Detection")
        axes[0, 1].axis('off')
        
        # Blobs
        blob_img = visualize_blobs(self.image, self.blobs)
        axes[1, 0].imshow(blob_img)
        axes[1, 0].set_title(f"Blobs ({len(self.blobs)})")
        axes[1, 0].axis('off')
        
        # Clusters
        cluster_img = np.zeros((*self.image.shape, 3))
        if self.clusters:
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.clusters)))
            for cluster_id, (cluster, color) in enumerate(zip(self.clusters, colors)):
                for pattern in cluster:
                    for blob in pattern.blobs:
                        for y, x in blob.pixels:
                            cluster_img[y, x] = color[:3]
        
        axes[1, 1].imshow(cluster_img)
        axes[1, 1].set_title(f"Pattern Clusters ({len(self.clusters)})")
        axes[1, 1].axis('off')
        
        plt.suptitle("CogAlg Processing Pipeline", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def get_summary(self):
        """Get a text summary of the results."""
        return {
            'image_shape': self.image.shape,
            'edge_pixels': np.sum(self.edges['edges']),
            'blob_count': len(self.blobs),
            'pattern_count': len(self.patterns),
            'cluster_count': len(self.clusters),
            'average_blob_size': np.mean([b.area for b in self.blobs]) if self.blobs else 0,
            'largest_blob_size': max([b.area for b in self.blobs]) if self.blobs else 0,
            'patterns_per_cluster': [len(c) for c in self.clusters]
        }


def process_image(image: Union[str, np.ndarray], 
                 edge_threshold: float = 30,
                 blob_threshold: float = 20,
                 cluster_threshold: float = 0.7,
                 verbose: bool = False) -> CogAlgResult:
    """
    Process an image through the CogAlg pipeline.
    
    Args:
        image: Path to image file or numpy array
        edge_threshold: Threshold for edge detection
        blob_threshold: Threshold for blob segmentation
        cluster_threshold: Similarity threshold for clustering
        verbose: Print processing steps
    
    Returns:
        CogAlgResult object containing all results
    """
    # Load image if path is provided
    if isinstance(image, str):
        import cv2
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image}")
    
    if verbose:
        print("CogAlg Processing Pipeline")
        print("=" * 40)
        print(f"Image shape: {image.shape}")
    
    # Step 1: Edge Detection
    if verbose:
        print("\n1. Detecting edges...")
    edges = detect_edges(image, threshold=edge_threshold)
    edge_count = np.sum(edges['edges'])
    if verbose:
        print(f"   Found {edge_count} edge pixels")
    
    # Step 2: Blob Detection
    if verbose:
        print("\n2. Finding blobs...")
    blobs = find_blobs(image, gradient=edges['gradient'], threshold=blob_threshold)
    if verbose:
        print(f"   Found {len(blobs)} blobs")
        if blobs:
            areas = [b.area for b in blobs]
            print(f"   Blob sizes: {min(areas)} - {max(areas)} pixels")
    
    # Step 3: Pattern Creation
    if verbose:
        print("\n3. Creating patterns...")
    patterns = [Pattern(id=i, blobs=[blob], features={}) 
                for i, blob in enumerate(blobs)]
    if verbose:
        print(f"   Created {len(patterns)} patterns")
    
    # Step 4: Pattern Clustering
    if verbose:
        print("\n4. Clustering patterns...")
    clusters = cluster_patterns(patterns, threshold=cluster_threshold)
    if verbose:
        print(f"   Formed {len(clusters)} clusters")
        for i, cluster in enumerate(clusters):
            if len(cluster) > 1:
                print(f"   Cluster {i}: {len(cluster)} similar patterns")
    
    # Create result object
    result = CogAlgResult(
        image=image,
        edges=edges,
        blobs=blobs,
        patterns=patterns,
        clusters=clusters
    )
    
    if verbose:
        print("\nProcessing complete!")
        summary = result.get_summary()
        print(f"\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    return result


def quick_demo():
    """Run a quick demonstration of CogAlg."""
    print("CogAlg Quick Demo")
    print("=" * 40)
    
    # Create a simple test image
    image = np.ones((40, 60)) * 128
    
    # Add some shapes
    image[5:15, 5:15] = 50      # Dark square
    image[5:15, 25:35] = 50     # Another dark square
    image[25:35, 15:25] = 200   # Bright square
    image[25:30, 40:55] = 200   # Bright rectangle
    
    # Process the image
    result = process_image(image, verbose=True)
    
    # Show results
    result.show_all()
    
    return result


# Convenience functions
def load_and_process(image_path: str, **kwargs) -> CogAlgResult:
    """Load an image file and process it."""
    return process_image(image_path, **kwargs)


def compare_images(image1: Union[str, np.ndarray], 
                  image2: Union[str, np.ndarray],
                  **kwargs) -> dict:
    """
    Compare patterns found in two images.
    
    Returns:
        Dictionary with comparison results
    """
    # Process both images
    result1 = process_image(image1, **kwargs)
    result2 = process_image(image2, **kwargs)
    
    # Compare all patterns between images
    matches = []
    for p1 in result1.patterns:
        for p2 in result2.patterns:
            match = compare_patterns(p1, p2)
            if match.similarity > 0.5:  # Significant similarity
                matches.append({
                    'pattern1_id': p1.id,
                    'pattern2_id': p2.id,
                    'similarity': match.similarity,
                    'match_type': match.match_type
                })
    
    return {
        'image1_patterns': len(result1.patterns),
        'image2_patterns': len(result2.patterns),
        'matches': matches,
        'match_count': len(matches)
    }


if __name__ == "__main__":
    # Run the demo
    result = quick_demo()
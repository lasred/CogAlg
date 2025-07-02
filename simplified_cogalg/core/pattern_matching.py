"""
Simple Pattern Matching

This module compares patterns (blobs) to find similarities and differences.
Like comparing shapes to see which ones match.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .blob_detection import Blob


@dataclass
class Pattern:
    """
    A pattern is a higher-level representation of image features.
    It can be a single blob or a group of related blobs.
    """
    id: int
    blobs: List[Blob]
    features: dict  # Extracted features for comparison
    
    @property
    def area(self):
        return sum(blob.area for blob in self.blobs)
    
    @property
    def intensity(self):
        if not self.blobs:
            return 0
        total_intensity = sum(blob.intensity * blob.area for blob in self.blobs)
        return total_intensity / self.area
    
    def __repr__(self):
        return f"Pattern(id={self.id}, blobs={len(self.blobs)}, area={self.area})"


@dataclass
class Match:
    """
    Represents a match between two patterns.
    """
    pattern1: Pattern
    pattern2: Pattern
    similarity: float  # 0-1, higher is more similar
    match_type: str   # 'shape', 'intensity', 'position', etc.
    details: dict     # Detailed comparison results


def compare_patterns(pattern1: Pattern, pattern2: Pattern, 
                    compare_shape=True, compare_intensity=True, 
                    compare_position=True) -> Match:
    """
    Compare two patterns and calculate their similarity.
    
    This is a simplified version of CogAlg's cross-comparison.
    
    Args:
        pattern1, pattern2: Patterns to compare
        compare_shape: Whether to compare shapes
        compare_intensity: Whether to compare intensities
        compare_position: Whether to compare positions
    
    Returns:
        Match object with similarity score and details
    """
    similarities = []
    details = {}
    
    # Compare intensity (average pixel value)
    if compare_intensity:
        intensity_sim = _compare_intensity(pattern1, pattern2)
        similarities.append(intensity_sim)
        details['intensity_similarity'] = intensity_sim
    
    # Compare shape (area and aspect ratio)
    if compare_shape:
        shape_sim = _compare_shape(pattern1, pattern2)
        similarities.append(shape_sim)
        details['shape_similarity'] = shape_sim
    
    # Compare position (distance between centers)
    if compare_position:
        position_sim = _compare_position(pattern1, pattern2)
        similarities.append(position_sim)
        details['position_similarity'] = position_sim
    
    # Overall similarity is average of all comparisons
    overall_similarity = np.mean(similarities) if similarities else 0
    
    # Determine match type (which aspect matches best)
    if similarities:
        match_types = ['intensity', 'shape', 'position']
        active_types = [mt for mt, active in zip(match_types, 
                       [compare_intensity, compare_shape, compare_position]) if active]
        best_match_idx = np.argmax(similarities)
        match_type = active_types[best_match_idx]
    else:
        match_type = 'none'
    
    return Match(
        pattern1=pattern1,
        pattern2=pattern2,
        similarity=overall_similarity,
        match_type=match_type,
        details=details
    )


def _compare_intensity(pattern1: Pattern, pattern2: Pattern) -> float:
    """
    Compare average intensities of patterns.
    
    Returns:
        Similarity score (0-1)
    """
    # Calculate normalized difference
    i1, i2 = pattern1.intensity, pattern2.intensity
    if max(i1, i2) == 0:
        return 1.0 if i1 == i2 else 0.0
    
    # Similarity based on relative difference
    diff = abs(i1 - i2)
    similarity = 1.0 - (diff / max(i1, i2))
    
    return max(0, similarity)


def _compare_shape(pattern1: Pattern, pattern2: Pattern) -> float:
    """
    Compare shapes of patterns based on area and bounding box.
    
    Returns:
        Similarity score (0-1)
    """
    # Compare areas
    area1, area2 = pattern1.area, pattern2.area
    if max(area1, area2) == 0:
        return 1.0
    area_similarity = min(area1, area2) / max(area1, area2)
    
    # Compare aspect ratios if patterns have single blobs
    if len(pattern1.blobs) == 1 and len(pattern2.blobs) == 1:
        aspect1 = _get_aspect_ratio(pattern1.blobs[0])
        aspect2 = _get_aspect_ratio(pattern2.blobs[0])
        
        if max(aspect1, aspect2) > 0:
            aspect_similarity = min(aspect1, aspect2) / max(aspect1, aspect2)
        else:
            aspect_similarity = 1.0
        
        # Average of area and aspect ratio similarity
        return (area_similarity + aspect_similarity) / 2
    
    return area_similarity


def _compare_position(pattern1: Pattern, pattern2: Pattern, max_distance=100) -> float:
    """
    Compare positions of patterns.
    
    Args:
        max_distance: Maximum distance for 0 similarity
    
    Returns:
        Similarity score (0-1)
    """
    # Calculate pattern centers
    center1 = _get_pattern_center(pattern1)
    center2 = _get_pattern_center(pattern2)
    
    # Calculate Euclidean distance
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    # Convert distance to similarity
    similarity = max(0, 1 - (distance / max_distance))
    
    return similarity


def _get_aspect_ratio(blob: Blob) -> float:
    """Calculate aspect ratio of a blob's bounding box."""
    min_y, min_x, max_y, max_x = blob.bounds
    height = max_y - min_y + 1
    width = max_x - min_x + 1
    return width / height if height > 0 else 1.0


def _get_pattern_center(pattern: Pattern) -> Tuple[float, float]:
    """Calculate center of mass for a pattern."""
    if not pattern.blobs:
        return (0, 0)
    
    total_y = sum(blob.center[0] * blob.area for blob in pattern.blobs)
    total_x = sum(blob.center[1] * blob.area for blob in pattern.blobs)
    total_area = pattern.area
    
    return (total_y / total_area, total_x / total_area)


def find_similar_patterns(query_pattern: Pattern, pattern_list: List[Pattern], 
                         threshold=0.7) -> List[Match]:
    """
    Find patterns similar to a query pattern.
    
    Args:
        query_pattern: Pattern to match against
        pattern_list: List of patterns to search
        threshold: Minimum similarity score
    
    Returns:
        List of Match objects above threshold, sorted by similarity
    """
    matches = []
    
    for pattern in pattern_list:
        if pattern.id != query_pattern.id:  # Don't match with self
            match = compare_patterns(query_pattern, pattern)
            if match.similarity >= threshold:
                matches.append(match)
    
    # Sort by similarity (highest first)
    matches.sort(key=lambda m: m.similarity, reverse=True)
    
    return matches


def cluster_patterns(patterns: List[Pattern], threshold=0.7) -> List[List[Pattern]]:
    """
    Group similar patterns into clusters.
    
    This is a simplified version of hierarchical clustering.
    
    Args:
        patterns: List of patterns to cluster
        threshold: Similarity threshold for grouping
    
    Returns:
        List of pattern clusters
    """
    if not patterns:
        return []
    
    # Initialize each pattern as its own cluster
    clusters = [[p] for p in patterns]
    
    # Keep merging most similar clusters until no more merges possible
    merged = True
    while merged:
        merged = False
        best_similarity = 0
        best_pair = None
        
        # Find most similar pair of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Compare cluster representatives (first pattern in each)
                similarity = compare_patterns(
                    Pattern(id=-1, blobs=sum([p.blobs for p in clusters[i]], [])),
                    Pattern(id=-1, blobs=sum([p.blobs for p in clusters[j]], []))
                ).similarity
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_pair = (i, j)
        
        # Merge best pair if found
        if best_pair:
            i, j = best_pair
            clusters[i].extend(clusters[j])
            clusters.pop(j)
            merged = True
    
    return clusters


# Example usage
if __name__ == "__main__":
    # Create some test patterns
    from .blob_detection import find_blobs
    
    # Test image with multiple regions
    test_image = np.array([
        [10, 10, 50, 50, 90, 90],
        [10, 10, 50, 50, 90, 90],
        [20, 20, 60, 60, 80, 80],
        [20, 20, 60, 60, 80, 80],
        [10, 10, 50, 50, 90, 90],
        [10, 10, 50, 50, 90, 90],
    ])
    
    # Find blobs and create patterns
    blobs = find_blobs(test_image)
    patterns = [Pattern(id=i, blobs=[blob], features={}) for i, blob in enumerate(blobs)]
    
    print(f"Found {len(patterns)} patterns")
    
    # Compare first two patterns
    if len(patterns) >= 2:
        match = compare_patterns(patterns[0], patterns[1])
        print(f"\nComparison between pattern 0 and 1:")
        print(f"  Similarity: {match.similarity:.2f}")
        print(f"  Match type: {match.match_type}")
        print(f"  Details: {match.details}")
    
    # Find similar patterns
    if patterns:
        similar = find_similar_patterns(patterns[0], patterns[1:], threshold=0.5)
        print(f"\nPatterns similar to pattern 0: {len(similar)}")
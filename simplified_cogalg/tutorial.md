# CogAlg Tutorial - Understanding Pattern Discovery

This tutorial will walk you through the core concepts of CogAlg using simple, understandable code.

## Table of Contents
1. [Introduction](#introduction)
2. [Step 1: Edge Detection](#step-1-edge-detection)
3. [Step 2: Blob Detection](#step-2-blob-detection)
4. [Step 3: Pattern Matching](#step-3-pattern-matching)
5. [Step 4: Hierarchical Patterns](#step-4-hierarchical-patterns)
6. [Complete Example](#complete-example)

## Introduction

CogAlg discovers patterns in images through a bottom-up process:
- Start with pixels
- Find edges (where things change)
- Group similar areas into blobs
- Compare blobs to find patterns
- Build hierarchies of patterns

Think of it like how you might recognize objects:
1. You see edges (outlines)
2. You group areas (sky, ground, objects)
3. You recognize patterns (this looks like a tree)
4. You understand relationships (forest = many trees)

## Step 1: Edge Detection

Edges are places where the image changes significantly. They often represent object boundaries.

```python
import numpy as np
from simplified_cogalg.core import detect_edges
import matplotlib.pyplot as plt

# Create a simple image with a square
image = np.zeros((10, 10))
image[3:7, 3:7] = 100  # White square on black background

# Detect edges
result = detect_edges(image, threshold=50)

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(result['gradient'], cmap='hot')
plt.title('Edge Strength (Gradient)')

plt.subplot(1, 3, 3)
plt.imshow(result['edges'], cmap='binary')
plt.title('Detected Edges')

plt.show()
```

**What's happening:**
- We calculate how much each pixel differs from its neighbors
- Large differences indicate edges
- The threshold determines sensitivity

## Step 2: Blob Detection

Blobs are connected regions of similar pixels. They represent coherent parts of the image.

```python
from simplified_cogalg.core import find_blobs, visualize_blobs

# Create an image with multiple regions
image = np.array([
    [10, 10, 10, 50, 50],
    [10, 10, 10, 50, 50],
    [10, 10, 10, 50, 50],
    [70, 70, 30, 30, 30],
    [70, 70, 30, 30, 30]
])

# Find blobs
blobs = find_blobs(image, threshold=20)

# Print information about each blob
for blob in blobs:
    print(f"Blob {blob.id}:")
    print(f"  Area: {blob.area} pixels")
    print(f"  Average intensity: {blob.intensity:.1f}")
    print(f"  Center: {blob.center}")
    print()

# Visualize blobs
blob_image = visualize_blobs(image, blobs)
plt.imshow(blob_image)
plt.title(f'Found {len(blobs)} blobs')
plt.show()
```

**What's happening:**
- We use "flood fill" to find connected pixels
- Similar pixels (within gradient threshold) group together
- Each blob stores its properties (size, intensity, location)

## Step 3: Pattern Matching

Now we compare blobs to find which ones are similar - these similarities are patterns.

```python
from simplified_cogalg.core import Pattern, compare_patterns

# Convert blobs to patterns
patterns = [Pattern(id=i, blobs=[blob], features={}) 
            for i, blob in enumerate(blobs)]

# Compare all pairs of patterns
print("Pattern Comparisons:")
for i in range(len(patterns)):
    for j in range(i+1, len(patterns)):
        match = compare_patterns(patterns[i], patterns[j])
        print(f"Pattern {i} vs Pattern {j}:")
        print(f"  Overall similarity: {match.similarity:.2%}")
        print(f"  Best match type: {match.match_type}")
        
# Find patterns similar to the first one
from simplified_cogalg.core import find_similar_patterns
similar = find_similar_patterns(patterns[0], patterns[1:], threshold=0.5)
print(f"\nPatterns similar to Pattern 0: {len(similar)}")
```

**What's happening:**
- We compare patterns by multiple features (intensity, shape, position)
- Similarity scores help identify matching patterns
- This is how the algorithm "recognizes" similar structures

## Step 4: Hierarchical Patterns

Patterns can contain other patterns, forming hierarchies. This is how complex structures are recognized.

```python
from simplified_cogalg.core import cluster_patterns

# Cluster similar patterns together
clusters = cluster_patterns(patterns, threshold=0.6)

print(f"Formed {len(clusters)} pattern clusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i}: contains {len(cluster)} patterns")
    areas = [p.area for p in cluster]
    print(f"  Total area: {sum(areas)} pixels")
    print(f"  Pattern IDs: {[p.id for p in cluster]}")
```

**What's happening:**
- Similar patterns group into higher-level patterns
- This creates a hierarchy (patterns of patterns)
- Complex objects are recognized as combinations of simpler patterns

## Complete Example

Let's put it all together with a real image:

```python
import cv2
from simplified_cogalg import process_image

# Load an image
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Process through all stages
edges = detect_edges(image, threshold=30)
blobs = find_blobs(image, edges['gradient'])
patterns = [Pattern(id=i, blobs=[blob], features={}) 
            for i, blob in enumerate(blobs)]

# Find and display pattern hierarchy
clusters = cluster_patterns(patterns, threshold=0.7)

print(f"Image Analysis Complete:")
print(f"  Edges found: {np.sum(edges['edges'])}")
print(f"  Blobs detected: {len(blobs)}")
print(f"  Pattern clusters: {len(clusters)}")

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0,0].imshow(image, cmap='gray')
axes[0,0].set_title('Original Image')

axes[0,1].imshow(edges['gradient'], cmap='hot')
axes[0,1].set_title('Edge Detection')

axes[1,0].imshow(visualize_blobs(image, blobs))
axes[1,0].set_title(f'{len(blobs)} Blobs Found')

# Show pattern clusters
cluster_img = np.zeros_like(image)
for i, cluster in enumerate(clusters):
    for pattern in cluster:
        for blob in pattern.blobs:
            for y, x in blob.pixels:
                cluster_img[y, x] = (i + 1) * 50

axes[1,1].imshow(cluster_img, cmap='tab20')
axes[1,1].set_title(f'{len(clusters)} Pattern Clusters')

plt.tight_layout()
plt.show()
```

## Key Concepts Summary

1. **Bottom-up Processing**: Start with pixels, build up to complex patterns
2. **Edge Detection**: Find boundaries and changes
3. **Blob Formation**: Group similar connected regions
4. **Pattern Matching**: Compare blobs to find similarities
5. **Hierarchical Structure**: Patterns contain other patterns

## Next Steps

1. Try with your own images
2. Adjust thresholds to see how it affects detection
3. Explore the pattern matching details
4. Implement your own comparison functions
5. Extend to video (temporal patterns)

## Tips for Understanding the Original CogAlg

The original CogAlg uses more complex versions of these concepts:
- Multiple types of comparison (range, angle, gradient)
- Recursive processing (patterns of patterns of patterns...)
- Complex data structures for efficiency
- Many optimization techniques

This simplified version captures the essential ideas while being much easier to understand and modify.
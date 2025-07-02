# Simplified CogAlg - A Beginner-Friendly Version

This is a simplified version of the CogAlg (Cognitive Algorithm) project, designed to make the core concepts more accessible to developers.

## What is CogAlg?

CogAlg is a pattern discovery algorithm that works by:
1. Finding edges in images (areas where pixels change)
2. Grouping similar areas into "blobs" (regions)
3. Finding patterns by comparing these blobs
4. Building hierarchies of patterns (patterns of patterns)

## Quick Start

```python
from simple_cogalg import process_image

# Process an image
results = process_image("image.jpg")

# View detected blobs
results.show_blobs()

# Get blob information
for blob in results.blobs:
    print(f"Blob area: {blob.area}, average intensity: {blob.intensity}")
```

## Project Structure

```
simplified_cogalg/
├── README.md                # This file
├── tutorial.md             # Step-by-step tutorial
├── core/                   # Core algorithms
│   ├── __init__.py
│   ├── edge_detection.py   # Simple edge detection
│   ├── blob_detection.py   # Blob segmentation
│   └── pattern_matching.py # Pattern comparison
├── examples/               # Working examples
│   ├── 01_basic_edges.py
│   ├── 02_find_blobs.py
│   ├── 03_compare_patterns.py
│   └── images/            # Sample images
├── utils/                  # Helper functions
│   ├── visualization.py    # Display results
│   └── data_structures.py  # Simple data classes
└── tests/                  # Unit tests
```

## Core Concepts Explained Simply

### 1. Edge Detection
- Compares neighboring pixels
- Finds where the image changes (edges)
- Like finding the outline of objects

### 2. Blob Formation
- Groups pixels that are similar
- Creates regions (blobs) of the image
- Like coloring by numbers

### 3. Pattern Matching
- Compares different blobs
- Finds which ones are similar
- Like finding matching puzzle pieces

### 4. Hierarchical Patterns
- Patterns can contain other patterns
- Like Russian nesting dolls
- Builds understanding from simple to complex

## Installation

```bash
pip install numpy opencv-python matplotlib
```

## Simple Example

```python
import numpy as np
from simplified_cogalg import SimpleBlob, find_blobs

# Create a simple test image
image = np.array([
    [10, 10, 50, 50],
    [10, 10, 50, 50],
    [20, 20, 60, 60],
    [20, 20, 60, 60]
])

# Find blobs (regions) in the image
blobs = find_blobs(image)

# Print what we found
for i, blob in enumerate(blobs):
    print(f"Blob {i}: {blob.pixel_count} pixels, average value: {blob.average_value}")
```

## Key Differences from Original CogAlg

1. **Clearer Names**: No cryptic abbreviations
2. **Simple Data Structures**: Basic classes instead of complex tuples
3. **Step-by-Step Processing**: Each step is a separate, understandable function
4. **Extensive Comments**: Every function is documented
5. **Visual Output**: Easy ways to see what's happening
6. **Modular Design**: Use only the parts you need

## Next Steps

1. Read the [tutorial.md](tutorial.md) for a step-by-step guide
2. Run the examples in order (01, 02, 03...)
3. Try with your own images
4. Explore the core algorithms
5. Contribute improvements!

## Original Project

This is based on Boris Kazachenko's [CogAlg](https://github.com/boris-kz/CogAlg) project.
The original implements a much more sophisticated version of these concepts.
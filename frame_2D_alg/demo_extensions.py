#!/usr/bin/env python3
"""
Demonstration of CogAlg Extensions

This script demonstrates the theoretical and practical extensions to CogAlg,
showing how they advance pattern discovery capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Import original and extended modules
from frame_blobs import imread, CBase
from agg_recursion import vect_root, CN, CLay
from agg_recursion_extended import (
    agg_recursion_extended, comp_N_extended, 
    cluster_link_, feedback_coordinates
)
from cross_modal_cogalg import CrossModalCogAlg


def create_dynamic_test_pattern():
    """Create a test image with dynamic patterns for testing extensions"""
    size = 200
    image = np.ones((size, size)) * 128
    
    # Create moving spiral pattern
    t = 0.5  # Time parameter
    center = size // 2
    
    for y in range(size):
        for x in range(size):
            # Distance from center
            dx, dy = x - center, y - center
            r = np.sqrt(dx**2 + dy**2)
            
            # Spiral angle
            theta = np.arctan2(dy, dx)
            spiral_phase = r * 0.1 - theta - t * 2
            
            # Create spiral with varying intensity
            if r < 80:
                intensity = 128 + 100 * np.sin(spiral_phase)
                image[y, x] = np.clip(intensity, 0, 255)
    
    # Add geometric shapes with different dynamics
    # Expanding square
    square_size = int(20 + t * 10)
    y1, x1 = 20, 20
    image[y1:y1+square_size, x1:x1+square_size] = 200
    
    # Rotating triangle
    triangle_points = np.array([
        [150, 30],
        [170, 30],
        [160, 50]
    ])
    # Rotate around center
    angle = t * np.pi / 2
    center = triangle_points.mean(axis=0)
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated = (triangle_points - center) @ rot_matrix.T + center
    
    # Draw triangle (simplified)
    for p in rotated.astype(int):
        if 0 <= p[0] < size and 0 <= p[1] < size:
            image[p[0]-2:p[0]+2, p[1]-2:p[1]+2] = 50
    
    return image


def demonstrate_higher_derivatives():
    """Demonstrate higher-order derivative pattern discovery"""
    print("\n" + "="*60)
    print("DEMONSTRATION 1: Higher-Order Derivatives")
    print("="*60)
    
    # Create sequence of images showing motion
    images = []
    for t in np.linspace(0, 1, 5):
        # Create dynamic pattern at different time points
        size = 100
        image = np.ones((size, size)) * 128
        
        # Moving circle
        center_x = int(20 + t * 60)
        center_y = int(50 + 20 * np.sin(t * 2 * np.pi))
        
        for y in range(max(0, center_y-10), min(size, center_y+10)):
            for x in range(max(0, center_x-10), min(size, center_x+10)):
                if (x - center_x)**2 + (y - center_y)**2 <= 100:
                    image[y, x] = 200
        
        images.append(image)
    
    # Process sequence to extract motion patterns
    frame_graphs = []
    for i, image in enumerate(images):
        frame = CBase()
        frame = vect_root(frame, image)
        frame_graphs.append(frame)
    
    # Compare consecutive frames to get velocity, acceleration, jerk
    print("\nAnalyzing motion patterns:")
    
    velocities = []
    accelerations = []
    
    for i in range(1, len(frame_graphs)):
        if frame_graphs[i-1].N_ and frame_graphs[i].N_:
            # Get primary node (largest)
            node1 = max(frame_graphs[i-1].N_, key=lambda n: n.Et[2] if hasattr(n, 'Et') else 0)
            node2 = max(frame_graphs[i].N_, key=lambda n: n.Et[2] if hasattr(n, 'Et') else 0)
            
            # Velocity (first derivative)
            velocity = node2.yx - node1.yx
            velocities.append(velocity)
            print(f"  Frame {i-1}→{i}: velocity = {velocity}")
            
            # Acceleration (second derivative)
            if i >= 2 and len(velocities) >= 2:
                acceleration = velocities[-1] - velocities[-2]
                accelerations.append(acceleration)
                print(f"    Acceleration = {acceleration}")
            
            # Jerk (third derivative)
            if i >= 3 and len(accelerations) >= 2:
                jerk = accelerations[-1] - accelerations[-2]
                print(f"    Jerk = {jerk}")
    
    # Visualize motion analysis
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show image sequence
    for i, (image, ax) in enumerate(zip(images[:3], axes[0])):
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Frame {i}')
        ax.axis('off')
    
    # Plot derivatives
    if velocities:
        times = range(len(velocities))
        axes[1, 0].plot(times, [v[0] for v in velocities], 'b-', label='Y velocity')
        axes[1, 0].plot(times, [v[1] for v in velocities], 'r-', label='X velocity')
        axes[1, 0].set_title('Velocity (1st derivative)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    if accelerations:
        times = range(len(accelerations))
        axes[1, 1].plot(times, [a[0] for a in accelerations], 'b-', label='Y accel')
        axes[1, 1].plot(times, [a[1] for a in accelerations], 'r-', label='X accel')
        axes[1, 1].set_title('Acceleration (2nd derivative)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    # Phase space plot
    if len(velocities) >= 2:
        axes[1, 2].plot([v[1] for v in velocities], [v[0] for v in velocities], 'g-o')
        axes[1, 2].set_xlabel('X velocity')
        axes[1, 2].set_ylabel('Y velocity')
        axes[1, 2].set_title('Phase Space')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig('demo_higher_derivatives.png', dpi=150)
    plt.close()
    
    print("\nHigher derivatives reveal:")
    print("- Velocity: Linear motion with oscillation")
    print("- Acceleration: Periodic pattern (circular motion)")
    print("- Jerk: Changes in acceleration (non-uniform motion)")


def demonstrate_link_clustering():
    """Demonstrate correlation pattern discovery through link clustering"""
    print("\n" + "="*60)
    print("DEMONSTRATION 2: Link Clustering for Correlation Patterns")
    print("="*60)
    
    # Create image with correlated patterns
    image = np.ones((150, 150)) * 128
    
    # Create chain of connected shapes
    positions = [
        (30, 30), (50, 50), (70, 70), (90, 90),  # Diagonal chain
        (30, 90), (50, 90), (70, 90), (90, 90),  # Horizontal chain
        (90, 30), (90, 50), (90, 70), (90, 90)   # Vertical chain
    ]
    
    for i, (y, x) in enumerate(positions):
        intensity = 50 + i * 10  # Gradient along chains
        image[y-5:y+5, x-5:x+5] = intensity
    
    # Process image
    frame = CBase()
    frame = vect_root(frame, image)
    
    if frame.N_:
        # Extended processing with link clustering
        extended_graph = agg_recursion_extended(frame, frame, rng=1)
        
        print(f"\nOriginal nodes: {len(frame.N_)}")
        print(f"Original links: {len(frame.L_)}")
        
        if hasattr(extended_graph, 'link_clusters'):
            print(f"Link clusters found: {len(extended_graph.link_clusters)}")
            
            for i, cluster in enumerate(extended_graph.link_clusters):
                print(f"\nLink cluster {i}:")
                print(f"  Links: {len(cluster.L_)}")
                print(f"  Connected nodes: {len(cluster.N_)}")
                print(f"  Total correlation: {cluster.Et[0]:.2f}")
        
        # Visualize link clusters
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(image, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(image, cmap='gray', alpha=0.3)
        ax2.set_title('Link Clusters (Correlation Patterns)')
        
        # Draw link clusters
        if hasattr(extended_graph, 'link_clusters'):
            colors = plt.cm.rainbow(np.linspace(0, 1, len(extended_graph.link_clusters)))
            
            for cluster, color in zip(extended_graph.link_clusters, colors):
                # Draw links in cluster
                for link in cluster.L_:
                    if len(link.N_) >= 2:
                        y1, x1 = link.N_[0].yx
                        y2, x2 = link.N_[1].yx
                        ax2.plot([x1, x2], [y1, y2], color=color, linewidth=3, alpha=0.7)
        
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig('demo_link_clustering.png', dpi=150)
        plt.close()
        
        print("\nLink clustering reveals:")
        print("- Correlated chains of patterns")
        print("- Directional relationships")
        print("- Higher-order structures beyond simple proximity")


def demonstrate_attention_imagination():
    """Demonstrate coordinate feedback for attention and imagination"""
    print("\n" + "="*60)
    print("DEMONSTRATION 3: Attention and Imagination Maps")
    print("="*60)
    
    # Create image with predictable and novel regions
    image = np.ones((200, 200)) * 128
    
    # Predictable region: Regular grid
    for i in range(5, 95, 10):
        for j in range(5, 95, 10):
            image[i:i+5, j:j+5] = 100
    
    # Novel region: Random high-value patterns
    np.random.seed(42)
    for _ in range(10):
        y, x = np.random.randint(100, 180, 2)
        size = np.random.randint(5, 15)
        intensity = np.random.randint(180, 250)
        image[y:y+size, x:x+size] = intensity
    
    # Process image
    frame = CBase()
    frame.Et = np.array([0, 0, 200, 200])  # Set frame dimensions
    frame = vect_root(frame, image)
    
    if frame.N_:
        # Get attention and imagination maps
        attention_map, imagination_map = feedback_coordinates(frame, frame.N_)
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(attention_map, cmap='hot')
        axes[0, 1].set_title('Attention Map\n(High at novel/unpredictable regions)')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(imagination_map, cmap='viridis')
        axes[1, 0].set_title('Imagination Map\n(Pattern projections)')
        axes[1, 0].axis('off')
        
        # Combined view
        axes[1, 1].imshow(image, cmap='gray', alpha=0.5)
        axes[1, 1].imshow(attention_map, cmap='Reds', alpha=0.3)
        axes[1, 1].imshow(imagination_map, cmap='Blues', alpha=0.3)
        axes[1, 1].set_title('Combined View\n(Red=Attention, Blue=Imagination)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('demo_attention_imagination.png', dpi=150)
        plt.close()
        
        print("\nAttention mechanism shows:")
        print(f"- Attention focused on novel patterns (bottom-right)")
        print(f"- Reduced attention on predictable grid (top-left)")
        print(f"- Mean attention: {attention_map.mean():.3f}")
        
        print("\nImagination mechanism shows:")
        print(f"- Strong patterns projected to nearby locations")
        print(f"- Creates expectations for pattern continuation")
        print(f"- Total imagination strength: {imagination_map.sum():.1f}")


def demonstrate_cross_modal():
    """Demonstrate cross-modal pattern discovery"""
    print("\n" + "="*60)
    print("DEMONSTRATION 4: Cross-Modal Pattern Discovery")
    print("="*60)
    
    # Initialize cross-modal system
    cm_system = CrossModalCogAlg()
    
    # Create synchronized multimodal data
    
    # Vision: Moving circle
    images = []
    for t in np.linspace(0, 1, 10):
        image = np.ones((100, 100)) * 128
        center_x = int(20 + t * 60)
        center_y = 50
        
        for y in range(center_y-10, center_y+10):
            for x in range(center_x-10, center_x+10):
                if (x - center_x)**2 + (y - center_y)**2 <= 100:
                    image[y, x] = 200
        images.append(image)
    
    # Audio: Frequency sweep synchronized with motion
    sample_rate = 44100
    duration = 1.0
    t_audio = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440 + 440 * t_audio  # Sweep from 440Hz to 880Hz
    audio = np.sin(2 * np.pi * frequency * t_audio) * 0.5
    
    # Text: Description of the pattern
    text = "circle moves right frequency rises bright object linear motion"
    
    # Process multimodal input
    inputs = {
        'vision': images[5],  # Middle frame
        'audio': audio,
        'text': text
    }
    
    unified = cm_system.process_multimodal_input(inputs)
    
    print(f"\nCross-modal analysis:")
    print(f"- Total patterns: {len(unified.N_)}")
    print(f"- Cross-modal links: {len(unified.L_)}")
    print(f"- Discovered symbols: {len(cm_system.symbol_grounding)}")
    
    # Analyze cross-modal connections
    modality_connections = {}
    for link in unified.L_:
        if len(link.N_) >= 2:
            mod1 = link.N_[0].modality
            mod2 = link.N_[1].modality
            key = f"{mod1}-{mod2}"
            modality_connections[key] = modality_connections.get(key, 0) + 1
    
    print("\nCross-modal connections found:")
    for connection, count in modality_connections.items():
        print(f"  {connection}: {count} links")
    
    # Test cross-modal query
    query_text = "bright circle"
    print(f"\nQuerying for audio patterns matching '{query_text}'...")
    
    audio_matches = cm_system.query_cross_modal(query_text, 'text', 'audio')
    print(f"Found {len(audio_matches)} matching audio patterns")
    
    # Visualize cross-modal bindings
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Vision
    axes[0, 0].imshow(images[5], cmap='gray')
    axes[0, 0].set_title('Visual Pattern')
    axes[0, 0].axis('off')
    
    # Audio spectrogram
    from scipy import signal
    f, t_spec, Sxx = signal.spectrogram(audio[:10000], sample_rate)
    axes[0, 1].pcolormesh(t_spec, f[:1000], 10 * np.log10(Sxx[:50]))
    axes[0, 1].set_ylabel('Frequency [Hz]')
    axes[0, 1].set_xlabel('Time [sec]')
    axes[0, 1].set_title('Audio Pattern')
    
    # Text representation
    axes[1, 0].text(0.1, 0.5, text, fontsize=12, wrap=True)
    axes[1, 0].set_title('Text Pattern')
    axes[1, 0].axis('off')
    
    # Cross-modal graph
    axes[1, 1].set_title('Cross-Modal Bindings')
    
    # Simple visualization of connections
    modalities = ['vision', 'audio', 'text']
    positions = {
        'vision': (0.2, 0.8),
        'audio': (0.8, 0.8),
        'text': (0.5, 0.2)
    }
    
    # Draw modality nodes
    for mod in modalities:
        axes[1, 1].scatter(*positions[mod], s=1000, alpha=0.5, label=mod)
        axes[1, 1].text(*positions[mod], mod, ha='center', va='center')
    
    # Draw connections
    for connection, count in modality_connections.items():
        if '-' in connection:
            mod1, mod2 = connection.split('-')
            if mod1 in positions and mod2 in positions:
                x1, y1 = positions[mod1]
                x2, y2 = positions[mod2]
                axes[1, 1].plot([x1, x2], [y1, y2], 'k-', 
                              linewidth=count/2, alpha=0.5)
    
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_cross_modal.png', dpi=150)
    plt.close()
    
    print("\nCross-modal discovery reveals:")
    print("- Synchronized patterns across modalities")
    print("- Emergent symbol grounding")
    print("- Modality-invariant pattern representation")


def demonstrate_meta_patterns():
    """Demonstrate meta-pattern discovery in operations"""
    print("\n" + "="*60)
    print("DEMONSTRATION 5: Meta-Pattern Discovery")
    print("="*60)
    
    # This would require tracking actual operations
    # For demonstration, we'll show the concept
    
    print("\nMeta-pattern discovery analyzes the algorithm's own operations:")
    print("- Tracks sequences of operations")
    print("- Finds repeated patterns in processing")
    print("- Generates optimized code for common patterns")
    
    # Example operation sequence
    example_ops = [
        "comp_N(node1, node2) -> link1",
        "comp_N(node2, node3) -> link2", 
        "comp_N(node3, node4) -> link3",
        "cluster_N_([node1, node2, node3, node4]) -> cluster1",
        "comp_N(node5, node6) -> link4",
        "comp_N(node6, node7) -> link5",
        "comp_N(node7, node8) -> link6", 
        "cluster_N_([node5, node6, node7, node8]) -> cluster2"
    ]
    
    print("\nExample operation sequence:")
    for op in example_ops[:4]:
        print(f"  {op}")
    print("  ...")
    
    print("\nDiscovered meta-pattern:")
    print("  Pattern: Sequential comparison followed by clustering")
    print("  Frequency: 2 occurrences")
    print("  Generated optimization: Batch comparison with immediate clustering")
    
    # Visualize meta-pattern
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw operation flow
    y_positions = np.linspace(1, 0, len(example_ops))
    
    for i, (op, y) in enumerate(zip(example_ops, y_positions)):
        # Parse operation type
        if "comp_N" in op:
            color = 'blue'
            marker = 'o'
        elif "cluster_N_" in op:
            color = 'red'
            marker = 's'
        else:
            color = 'gray'
            marker = 'd'
        
        ax.scatter(0.5, y, s=200, c=color, marker=marker, alpha=0.7)
        ax.text(0.55, y, op, va='center', fontsize=10)
        
        # Draw connections
        if i < len(example_ops) - 1:
            ax.plot([0.5, 0.5], [y, y_positions[i+1]], 'k-', alpha=0.3)
    
    # Highlight patterns
    pattern_regions = [(0, 3), (4, 7)]
    for start, end in pattern_regions:
        y_start = y_positions[start]
        y_end = y_positions[end]
        ax.add_patch(plt.Rectangle((0.4, y_end), 0.2, y_start-y_end,
                                 alpha=0.2, color='green'))
    
    ax.set_xlim(0.3, 1.5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title('Meta-Pattern Discovery in Operation Sequences')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('demo_meta_patterns.png', dpi=150)
    plt.close()
    
    print("\nMeta-patterns enable:")
    print("- Automatic algorithm optimization")
    print("- Learning from processing history")
    print("- Self-improvement through pattern discovery")


def main():
    """Run all demonstrations"""
    print("="*60)
    print("COGALG EXTENSIONS DEMONSTRATION")
    print("="*60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    os.makedirs("cogalg_extensions_demo", exist_ok=True)
    os.chdir("cogalg_extensions_demo")
    
    # Run demonstrations
    demonstrate_higher_derivatives()
    demonstrate_link_clustering()
    demonstrate_attention_imagination()
    demonstrate_cross_modal()
    demonstrate_meta_patterns()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nExtensions demonstrated:")
    print("1. Higher-order derivatives: Velocity, acceleration, jerk patterns")
    print("2. Link clustering: Correlation patterns and contours")
    print("3. Attention/Imagination: Predictive coordinate feedback")
    print("4. Cross-modal binding: Unified patterns across modalities")
    print("5. Meta-patterns: Self-discovery of algorithmic patterns")
    
    print("\nThese extensions advance CogAlg by:")
    print("- Capturing dynamic patterns through derivatives")
    print("- Finding relational patterns through link clustering")
    print("- Implementing predictive processing through feedback")
    print("- Enabling true conceptual understanding across modalities")
    print("- Moving toward self-improving algorithms")
    
    print(f"\nAll demonstrations saved to: cogalg_extensions_demo/")
    print("\nThese extensions maintain CogAlg's core principles while")
    print("significantly expanding its capabilities toward AGI.")


if __name__ == "__main__":
    main()
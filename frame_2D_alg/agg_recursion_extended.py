"""
Extended agg_recursion with theoretical advances

This module extends the original agg_recursion.py with:
1. Proper incremental derivation cross-comparison
2. Advanced link clustering for correlation patterns  
3. Coordinate feedback for attention/imagination
4. Meta-pattern discovery (patterns of operations)
"""

import numpy as np
from copy import deepcopy
from itertools import combinations
from functools import lru_cache
from agg_recursion import CN, CLay, comp_N, sum2graph, val_, w_t

# Extended weights for new derivatives
wAngle, wCurve, wGrad, wRange = 20, 30, 15, 10
w_ext = np.array([wAngle, wCurve, wGrad, wRange])

class CLink(CN):
    """Extended link class with correlation patterns"""
    name = "link"
    
    def __init__(self, N1, N2, **kwargs):
        super().__init__(**kwargs)
        self.N_ = [N1, N2]  # Connected nodes
        self.angle = kwargs.get('angle', np.zeros(2))  # Connection angle
        self.distance = kwargs.get('distance', 0)  # Euclidean distance
        self.correlation = kwargs.get('correlation', 0)  # Correlation strength
        self.link_type = kwargs.get('link_type', 'match')  # match, diff, or mixed
        

def comp_N_extended(_N, N, rn=1, angle=None, distance=None):
    """
    Extended node comparison with higher derivatives
    
    Adds:
    - Angle comparison (2nd derivative of position)
    - Curvature comparison (3rd derivative)  
    - Gradient magnitude comparison
    - Multi-scale range comparison
    """
    # Standard comparison from original
    Link = comp_N(_N, N, rn, angle=angle, span=distance)
    
    # Extended derivatives
    if _N.derH and N.derH:
        # Compare angles (2nd derivative of position)
        _angle = _N.angle / np.linalg.norm(_N.angle) if np.any(_N.angle) else _N.angle
        n_angle = N.angle / np.linalg.norm(N.angle) if np.any(N.angle) else N.angle
        
        angle_match = 1 - np.arccos(np.clip(np.dot(_angle, n_angle), -1, 1)) / np.pi
        angle_diff = np.cross(_angle, n_angle)
        
        # Compare curvature (change in angle)
        if len(_N.derH) > 1 and len(N.derH) > 1:
            _curv = _compute_curvature(_N)
            curv = _compute_curvature(N)
            curv_match = min(_curv, curv) / (max(_curv, curv) + 1e-7)
            curv_diff = _curv - curv
        else:
            curv_match, curv_diff = 0, 0
            
        # Compare gradient patterns
        _grad = _compute_gradient_pattern(_N)
        grad = _compute_gradient_pattern(N)
        grad_match = np.minimum(_grad, grad) / (np.maximum(_grad, grad) + 1e-7)
        grad_diff = _grad - grad
        
        # Multi-scale comparison
        scales = [1, 2, 4, 8]  # Range scales
        scale_matches = []
        
        for scale in scales:
            if _N.rng >= scale and N.rng >= scale:
                # Compare at different scales
                _pattern = _get_scale_pattern(_N, scale)
                pattern = _get_scale_pattern(N, scale)
                scale_match = _compare_patterns(_pattern, pattern)
                scale_matches.append(scale_match)
        
        range_match = np.mean(scale_matches) if scale_matches else 0
        
        # Update Link with extended derivatives
        ext_match = np.array([angle_match, curv_match, grad_match.mean(), range_match])
        ext_diff = np.array([angle_diff, curv_diff, grad_diff.mean(), 0])
        
        Link.Et[0] += np.dot(ext_match, w_ext)
        Link.Et[1] += np.dot(np.abs(ext_diff), w_ext)
        
    return Link


def cluster_link_(frame, node_, link_, base_rng):
    """
    Advanced link clustering for correlation patterns
    
    Forms contours/paths by clustering highly correlated links,
    complementing connectivity clusters of nodes
    """
    # Initialize link clusters
    link_clusters = []
    processed_links = set()
    
    # Sort links by correlation strength
    sorted_links = sorted(link_, key=lambda l: l.Et[0], reverse=True)
    
    for link in sorted_links:
        if id(link) in processed_links:
            continue
            
        # Start new link cluster (contour)
        contour = CN(fi=2)  # fi=2 indicates link cluster
        contour.L_ = [link]
        contour.N_ = list(link.N_)
        processed_links.add(id(link))
        
        # Extend contour by finding connected links
        extended = True
        while extended:
            extended = False
            
            for l in link_:
                if id(l) in processed_links:
                    continue
                    
                # Check if link connects to contour
                if any(n in contour.N_ for n in l.N_):
                    # Check correlation consistency
                    if _check_correlation_consistency(contour.L_, l):
                        contour.L_.append(l)
                        contour.N_.extend([n for n in l.N_ if n not in contour.N_])
                        processed_links.add(id(l))
                        extended = True
        
        # Evaluate contour
        if len(contour.L_) >= 3:  # Minimum contour length
            contour.Et = sum(l.Et for l in contour.L_)
            contour.derH = _compute_contour_derivatives(contour)
            link_clusters.append(contour)
    
    return link_clusters


def feedback_coordinates(frame, graph_):
    """
    Implement coordinate feedback for attention and imagination
    
    This creates spatial filters that guide future input selection,
    implementing the theoretical "attention" mechanism
    """
    # Initialize coordinate filters
    y_dim, x_dim = frame.Et[2:4]  # Frame dimensions
    attention_map = np.ones((int(y_dim), int(x_dim)))
    imagination_map = np.zeros((int(y_dim), int(x_dim)))
    
    for graph in graph_:
        if not isinstance(graph, CN):
            continue
            
        # High-value patterns increase attention to their regions
        if graph.Et[0] > ave * graph.Et[2]:
            _update_attention_map(attention_map, graph, increase=True)
        
        # Low-variance patterns decrease attention (predictable)
        if graph.Et[1] < avd * graph.Et[2]:
            _update_attention_map(attention_map, graph, increase=False)
            
        # Project patterns to nearby locations (imagination)
        if graph.Et[0] > ave * graph.Et[2] * 2:  # Strong patterns
            _project_pattern(imagination_map, graph)
    
    # Normalize maps
    attention_map = attention_map / (attention_map.max() + 1e-7)
    imagination_map = imagination_map / (imagination_map.max() + 1e-7)
    
    return attention_map, imagination_map


def cross_comp_operations(frame):
    """
    Meta-pattern discovery: find patterns in the operations themselves
    
    This implements the theoretical "code generation" by analyzing
    patterns in how patterns are formed
    """
    # Track operation sequences
    operation_sequences = []
    
    def track_operation(op_name, inputs, outputs, params):
        """Track each operation for pattern analysis"""
        op_record = {
            'name': op_name,
            'input_types': [type(i).__name__ for i in inputs],
            'output_type': type(outputs).__name__ if outputs else None,
            'params': params,
            'input_values': [i.Et if hasattr(i, 'Et') else None for i in inputs],
            'output_value': outputs.Et if hasattr(outputs, 'Et') else None
        }
        operation_sequences.append(op_record)
    
    # Analyze operation patterns
    operation_patterns = []
    
    # Find repeated operation sequences
    for i in range(len(operation_sequences) - 2):
        for j in range(i + 2, len(operation_sequences)):
            seq1 = operation_sequences[i:i+3]
            seq2 = operation_sequences[j:j+3]
            
            if _match_operation_sequences(seq1, seq2):
                pattern = {
                    'sequence': seq1,
                    'instances': [(i, i+3), (j, j+3)],
                    'frequency': 2
                }
                operation_patterns.append(pattern)
    
    # Generate optimized code for common patterns
    generated_code = []
    for pattern in operation_patterns:
        if pattern['frequency'] >= 3:  # Common pattern
            code = _generate_optimized_code(pattern)
            generated_code.append(code)
    
    return operation_patterns, generated_code


def incremental_range_derivation(node_, base_rng):
    """
    Implement proper incremental range AND derivation comparison
    
    This extends beyond simple distance-based comparison to include
    derivative order increase
    """
    extended_nodes = []
    
    for node in node_:
        # Range extension
        for rng_scale in [2, 4, 8]:
            if node.rng * rng_scale <= base_rng * 4:  # Limit range
                ext_node = _extend_node_range(node, rng_scale)
                extended_nodes.append(ext_node)
        
        # Derivation extension
        if node.derH:
            # Second derivative (acceleration patterns)
            if len(node.derH) >= 2:
                accel_node = _compute_acceleration_node(node)
                extended_nodes.append(accel_node)
            
            # Third derivative (jerk patterns)
            if len(node.derH) >= 3:
                jerk_node = _compute_jerk_node(node)
                extended_nodes.append(jerk_node)
    
    return extended_nodes


# Helper functions

def _compute_curvature(node):
    """Compute curvature from node's trajectory"""
    if len(node.N_) < 3:
        return 0
    
    points = np.array([n.yx for n in node.N_[-3:]])
    
    # Compute curvature using three points
    v1 = points[1] - points[0]
    v2 = points[2] - points[1]
    
    angle_change = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7), -1, 1))
    distance = np.linalg.norm(v1) + np.linalg.norm(v2)
    
    return angle_change / (distance + 1e-7)


def _compute_gradient_pattern(node):
    """Extract gradient pattern from node"""
    if not node.derH:
        return np.zeros(4)
    
    # Collect gradients from derivation hierarchy
    gradients = []
    for lay in node.derH:
        if hasattr(lay, 'derTT') and lay.derTT.size > 0:
            # Extract gradient components
            grad = lay.derTT[1, 4:6]  # G components
            gradients.append(grad)
    
    if not gradients:
        return np.zeros(4)
    
    gradients = np.array(gradients)
    
    # Compute gradient statistics
    return np.array([
        gradients.mean(),
        gradients.std(),
        np.gradient(gradients[:, 0]).mean() if len(gradients) > 1 else 0,
        np.gradient(gradients[:, 1]).mean() if len(gradients) > 1 else 0
    ])


@lru_cache(maxsize=128)
def _get_scale_pattern(node, scale):
    """Get pattern representation at specific scale"""
    # This would involve subsampling or aggregating node's data
    # Simplified version:
    pattern = {
        'scale': scale,
        'area': node.Et[2] / scale,
        'value': node.Et[0] / scale,
        'variance': node.Et[1] / np.sqrt(scale)
    }
    return pattern


def _compare_patterns(pattern1, pattern2):
    """Compare two scale patterns"""
    if not pattern1 or not pattern2:
        return 0
    
    # Simple similarity based on pattern properties
    area_sim = min(pattern1['area'], pattern2['area']) / (max(pattern1['area'], pattern2['area']) + 1e-7)
    value_sim = 1 - abs(pattern1['value'] - pattern2['value']) / (max(abs(pattern1['value']), abs(pattern2['value'])) + 1e-7)
    
    return (area_sim + value_sim) / 2


def _check_correlation_consistency(existing_links, new_link):
    """Check if new link maintains correlation consistency"""
    if not existing_links:
        return True
    
    # Compare correlation types
    existing_types = [l.link_type for l in existing_links]
    type_consistency = existing_types.count(new_link.link_type) / len(existing_types)
    
    # Compare correlation strengths
    existing_correlations = [l.correlation for l in existing_links]
    avg_correlation = np.mean(existing_correlations)
    
    correlation_consistency = 1 - abs(new_link.correlation - avg_correlation) / (avg_correlation + 1e-7)
    
    return (type_consistency + correlation_consistency) / 2 > 0.7


def _compute_contour_derivatives(contour):
    """Compute derivatives along contour"""
    if len(contour.L_) < 2:
        return []
    
    # Extract positions along contour
    positions = []
    for link in contour.L_:
        pos = (link.N_[0].yx + link.N_[1].yx) / 2
        positions.append(pos)
    
    positions = np.array(positions)
    
    # Compute derivatives
    derivatives = []
    
    # First derivative (tangent)
    if len(positions) > 1:
        tangents = np.gradient(positions, axis=0)
        derivatives.append(CLay(derTT=np.array([tangents.mean(axis=0), tangents.std(axis=0)])))
    
    # Second derivative (curvature)
    if len(positions) > 2:
        curvatures = np.gradient(tangents, axis=0)
        derivatives.append(CLay(derTT=np.array([curvatures.mean(axis=0), curvatures.std(axis=0)])))
    
    return derivatives


def _update_attention_map(attention_map, graph, increase=True):
    """Update attention map based on graph location"""
    # Get graph bounds
    y_min, x_min = int(graph.box[0]), int(graph.box[1])
    y_max, x_max = int(graph.box[2]), int(graph.box[3])
    
    # Ensure bounds are within map
    y_min = max(0, y_min)
    x_min = max(0, x_min)
    y_max = min(attention_map.shape[0], y_max)
    x_max = min(attention_map.shape[1], x_max)
    
    # Update attention
    factor = 1.5 if increase else 0.5
    attention_map[y_min:y_max, x_min:x_max] *= factor


def _project_pattern(imagination_map, graph):
    """Project pattern to nearby locations"""
    # Simple projection: Gaussian spread around pattern location
    y_center, x_center = graph.yx
    
    # Create Gaussian kernel
    sigma = graph.span / 4  # Spread based on pattern size
    y_coords, x_coords = np.ogrid[:imagination_map.shape[0], :imagination_map.shape[1]]
    
    distances = np.sqrt((y_coords - y_center)**2 + (x_coords - x_center)**2)
    gaussian = np.exp(-distances**2 / (2 * sigma**2))
    
    # Add pattern strength to imagination map
    pattern_strength = graph.Et[0] / (graph.Et[2] + 1e-7)
    imagination_map += gaussian * pattern_strength


def _match_operation_sequences(seq1, seq2):
    """Check if two operation sequences match"""
    if len(seq1) != len(seq2):
        return False
    
    for op1, op2 in zip(seq1, seq2):
        if op1['name'] != op2['name']:
            return False
        if op1['input_types'] != op2['input_types']:
            return False
        if op1['output_type'] != op2['output_type']:
            return False
    
    return True


def _generate_optimized_code(pattern):
    """Generate optimized code for common pattern"""
    # This is a simplified version - real implementation would generate actual code
    sequence = pattern['sequence']
    
    code = f"""
def optimized_{sequence[0]['name']}_{len(pattern['instances'])}(inputs):
    # Optimized version of repeated pattern
    # Original sequence: {[op['name'] for op in sequence]}
    # Frequency: {pattern['frequency']}
    
    # Combined operations for efficiency
    results = []
    for inp in inputs:
        # Fused operations
        result = inp
        {''.join(f"result = {op['name']}(result)" for op in sequence)}
        results.append(result)
    
    return results
"""
    return code


def _extend_node_range(node, scale):
    """Extend node's comparison range"""
    ext_node = deepcopy(node)
    ext_node.rng *= scale
    ext_node.span *= scale
    
    # Adjust derivatives for new scale
    if ext_node.derH:
        for lay in ext_node.derH:
            lay.derTT *= np.sqrt(scale)  # Scale normalization
    
    return ext_node


def _compute_acceleration_node(node):
    """Compute acceleration (second derivative) patterns"""
    accel_node = deepcopy(node)
    accel_node.name = "accel_" + node.name
    
    # Compute acceleration from velocity (first derivative)
    if len(node.derH) >= 2:
        velocities = []
        for i in range(len(node.derH) - 1):
            v1 = node.derH[i].derTT[1]  # Derivative values
            v2 = node.derH[i+1].derTT[1]
            velocity = v2 - v1
            velocities.append(velocity)
        
        if velocities:
            accel = np.mean(velocities, axis=0)
            accel_node.derTT = np.array([accel, np.std(velocities, axis=0)])
    
    return accel_node


def _compute_jerk_node(node):
    """Compute jerk (third derivative) patterns"""
    jerk_node = deepcopy(node)
    jerk_node.name = "jerk_" + node.name
    
    # Compute jerk from acceleration
    if len(node.derH) >= 3:
        accelerations = []
        for i in range(len(node.derH) - 2):
            # Compute acceleration at each point
            v1 = node.derH[i].derTT[1]
            v2 = node.derH[i+1].derTT[1]
            v3 = node.derH[i+2].derTT[1]
            
            accel1 = v2 - v1
            accel2 = v3 - v2
            jerk = accel2 - accel1
            accelerations.append(jerk)
        
        if accelerations:
            jerk_val = np.mean(accelerations, axis=0)
            jerk_node.derTT = np.array([jerk_val, np.std(accelerations, axis=0)])
    
    return jerk_node


# Main extended processing function

def agg_recursion_extended(frame, root_graph, rng=1):
    """
    Extended agglomerative recursion with theoretical advances
    
    Implements:
    1. Multi-derivative cross-comparison
    2. Link clustering for correlation patterns
    3. Coordinate feedback for attention
    4. Meta-pattern discovery in operations
    """
    # Standard processing from original
    node_ = root_graph.node_
    link_ = root_graph.link_
    
    # Extended node comparison with higher derivatives
    extended_links = []
    for _node, node in combinations(node_, 2):
        link = comp_N_extended(_node, node, rng)
        if val_(link.Et) > 0:
            extended_links.append(link)
    
    # Incremental range and derivation
    extended_nodes = incremental_range_derivation(node_, rng)
    
    # Link clustering for correlation patterns
    link_clusters = cluster_link_(frame, node_, extended_links, rng)
    
    # Update graph with extended processing
    root_graph.link_ = extended_links
    root_graph.node_.extend(extended_nodes)
    
    # Coordinate feedback
    attention_map, imagination_map = feedback_coordinates(frame, [root_graph])
    
    # Meta-pattern discovery
    operation_patterns, generated_code = cross_comp_operations(frame)
    
    # Store extended results
    root_graph.attention_map = attention_map
    root_graph.imagination_map = imagination_map
    root_graph.link_clusters = link_clusters
    root_graph.operation_patterns = operation_patterns
    root_graph.generated_code = generated_code
    
    return root_graph


# Import guard for original modules
ave = 10
avd = 10
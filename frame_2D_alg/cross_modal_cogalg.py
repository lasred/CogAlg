"""
Cross-Modal CogAlg: Unified Pattern Discovery Across Modalities

This module extends CogAlg to discover patterns across different data modalities
(vision, audio, text, sensor data) using the same fundamental principles.

Key innovations:
1. Modality-agnostic pattern representation
2. Cross-modal pattern matching
3. Emergent symbol grounding
4. Temporal pattern integration
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

from agg_recursion import CN, CLay, comp_N, val_
from agg_recursion_extended import comp_N_extended


@dataclass
class ModalityDescriptor:
    """Describes properties of a data modality"""
    name: str
    dimensions: List[str]  # e.g., ['y', 'x'] for vision, ['time', 'frequency'] for audio
    resolution: List[float]  # Resolution in each dimension
    value_range: Tuple[float, float]  # Expected value range
    comparison_metric: str  # Primary comparison metric


class ModalityAdapter(ABC):
    """Abstract base for modality-specific adaptations"""
    
    @abstractmethod
    def extract_primitives(self, data: np.ndarray) -> List[CN]:
        """Extract primitive patterns from raw modality data"""
        pass
    
    @abstractmethod
    def compute_derivatives(self, primitives: List[CN]) -> List[CN]:
        """Compute modality-specific derivatives"""
        pass
    
    @abstractmethod
    def normalize_pattern(self, pattern: CN) -> CN:
        """Normalize pattern to common representation"""
        pass


class VisionAdapter(ModalityAdapter):
    """Adapter for visual data (images, video)"""
    
    def extract_primitives(self, image: np.ndarray) -> List[CN]:
        """Extract visual primitives (edges, blobs, textures)"""
        from frame_blobs import frame_blobs_root
        
        # Use existing visual processing
        frame = frame_blobs_root(image)
        
        # Convert blobs to normalized CN format
        primitives = []
        for blob in frame.blob_:
            node = CN()
            node.yx = np.array([blob.y, blob.x])
            node.box = blob.box
            node.Et = np.array([blob.M, blob.D, blob.S])
            
            # Visual-specific attributes
            node.modality = 'vision'
            node.visual_features = {
                'gradient': blob.G,
                'angle': blob.A,
                'intensity': blob.I,
                'area': blob.S
            }
            primitives.append(node)
        
        return primitives
    
    def compute_derivatives(self, primitives: List[CN]) -> List[CN]:
        """Compute visual derivatives (motion, deformation)"""
        derivatives = []
        
        for i in range(1, len(primitives)):
            prev, curr = primitives[i-1], primitives[i]
            
            # Motion vector
            motion = curr.yx - prev.yx
            
            # Shape change
            shape_change = (curr.box[2:] - curr.box[:2]) - (prev.box[2:] - prev.box[:2])
            
            # Create derivative node
            deriv = CN()
            deriv.modality = 'vision_derivative'
            deriv.derTT = np.array([motion, shape_change])
            derivatives.append(deriv)
        
        return derivatives
    
    def normalize_pattern(self, pattern: CN) -> CN:
        """Normalize visual pattern to common space"""
        norm = CN()
        
        # Spatial normalization
        norm.position = pattern.yx / np.array([480, 640])  # Normalize to [0,1]
        
        # Feature normalization
        if hasattr(pattern, 'visual_features'):
            features = pattern.visual_features
            norm.features = np.array([
                features['intensity'] / 255,
                features['gradient'] / 100,
                np.cos(features['angle']),
                np.sin(features['angle']),
                features['area'] / 1000
            ])
        
        norm.modality = pattern.modality
        return norm


class AudioAdapter(ModalityAdapter):
    """Adapter for audio data"""
    
    def extract_primitives(self, audio: np.ndarray, sample_rate: int = 44100) -> List[CN]:
        """Extract audio primitives (onsets, pitch, timbre)"""
        primitives = []
        
        # Simple onset detection using energy
        window_size = 1024
        hop_size = 512
        
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i+window_size]
            
            # Energy
            energy = np.sum(window**2)
            
            # Spectral features (simplified)
            fft = np.fft.rfft(window)
            spectrum = np.abs(fft)
            
            # Spectral centroid (brightness)
            freqs = np.fft.rfftfreq(window_size, 1/sample_rate)
            centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-7)
            
            # Zero crossing rate (texture)
            zcr = np.sum(np.diff(np.sign(window)) != 0) / window_size
            
            # Create primitive node
            node = CN()
            node.modality = 'audio'
            node.yx = np.array([i / sample_rate, 0])  # Time position
            node.audio_features = {
                'energy': energy,
                'centroid': centroid,
                'zcr': zcr,
                'spectrum': spectrum[:128]  # Keep first 128 bins
            }
            node.Et = np.array([energy, centroid, zcr])
            
            primitives.append(node)
        
        return primitives
    
    def compute_derivatives(self, primitives: List[CN]) -> List[CN]:
        """Compute audio derivatives (pitch change, rhythm)"""
        derivatives = []
        
        for i in range(1, len(primitives)):
            prev, curr = primitives[i-1], primitives[i]
            
            # Energy change (onset strength)
            energy_change = curr.audio_features['energy'] - prev.audio_features['energy']
            
            # Pitch change
            pitch_change = curr.audio_features['centroid'] - prev.audio_features['centroid']
            
            # Rhythmic interval
            time_interval = curr.yx[0] - prev.yx[0]
            
            deriv = CN()
            deriv.modality = 'audio_derivative'
            deriv.derTT = np.array([energy_change, pitch_change, time_interval])
            derivatives.append(deriv)
        
        return derivatives
    
    def normalize_pattern(self, pattern: CN) -> CN:
        """Normalize audio pattern"""
        norm = CN()
        
        # Temporal normalization
        norm.position = np.array([pattern.yx[0] / 10.0, 0])  # Normalize time to ~10s
        
        # Feature normalization
        if hasattr(pattern, 'audio_features'):
            features = pattern.audio_features
            norm.features = np.array([
                np.log10(features['energy'] + 1) / 10,  # Log energy
                features['centroid'] / 20000,  # Normalize frequency
                features['zcr'],  # Already normalized
            ])
        
        norm.modality = pattern.modality
        return norm


class TextAdapter(ModalityAdapter):
    """Adapter for text/symbolic data"""
    
    def __init__(self):
        # Simple word embedding (in practice, use proper embeddings)
        self.word_vectors = {}
        self.vector_dim = 50
    
    def extract_primitives(self, text: str) -> List[CN]:
        """Extract text primitives (words, phrases, concepts)"""
        words = text.lower().split()
        primitives = []
        
        for i, word in enumerate(words):
            node = CN()
            node.modality = 'text'
            node.yx = np.array([0, i])  # Sequential position
            
            # Get or create word vector
            if word not in self.word_vectors:
                # Random initialization (in practice, use trained embeddings)
                self.word_vectors[word] = np.random.randn(self.vector_dim) * 0.1
            
            node.text_features = {
                'word': word,
                'vector': self.word_vectors[word],
                'length': len(word),
                'position': i / len(words)
            }
            
            node.Et = np.array([
                np.linalg.norm(self.word_vectors[word]),  # Vector magnitude
                len(word),  # Word length
                i  # Position
            ])
            
            primitives.append(node)
        
        return primitives
    
    def compute_derivatives(self, primitives: List[CN]) -> List[CN]:
        """Compute text derivatives (semantic change, syntax)"""
        derivatives = []
        
        for i in range(1, len(primitives)):
            prev, curr = primitives[i-1], primitives[i]
            
            # Semantic distance
            vec1 = prev.text_features['vector']
            vec2 = curr.text_features['vector']
            semantic_dist = np.linalg.norm(vec2 - vec1)
            
            # Syntactic features (simplified)
            length_change = curr.text_features['length'] - prev.text_features['length']
            
            deriv = CN()
            deriv.modality = 'text_derivative'
            deriv.derTT = np.array([semantic_dist, length_change])
            derivatives.append(deriv)
        
        return derivatives
    
    def normalize_pattern(self, pattern: CN) -> CN:
        """Normalize text pattern"""
        norm = CN()
        
        # Position normalization
        norm.position = np.array([0, pattern.text_features['position']])
        
        # Feature normalization
        if hasattr(pattern, 'text_features'):
            # Use first few components of word vector
            norm.features = pattern.text_features['vector'][:5] / 10
        
        norm.modality = pattern.modality
        return norm


class CrossModalCogAlg:
    """Main cross-modal pattern discovery system"""
    
    def __init__(self):
        self.adapters = {
            'vision': VisionAdapter(),
            'audio': AudioAdapter(),
            'text': TextAdapter()
        }
        
        self.modality_descriptors = {
            'vision': ModalityDescriptor(
                name='vision',
                dimensions=['y', 'x'],
                resolution=[1.0, 1.0],
                value_range=(0, 255),
                comparison_metric='euclidean'
            ),
            'audio': ModalityDescriptor(
                name='audio',
                dimensions=['time', 'frequency'],
                resolution=[1/44100, 1.0],
                value_range=(-1, 1),
                comparison_metric='spectral'
            ),
            'text': ModalityDescriptor(
                name='text',
                dimensions=['sequence', 'semantic'],
                resolution=[1.0, 1.0],
                value_range=(0, 1),
                comparison_metric='cosine'
            )
        }
        
        self.cross_modal_patterns = []
        self.symbol_grounding = {}  # Maps abstract patterns to concrete instances
    
    def process_multimodal_input(self, inputs: Dict[str, Any]) -> CN:
        """
        Process inputs from multiple modalities simultaneously
        
        Args:
            inputs: Dict mapping modality names to data
        
        Returns:
            Unified cross-modal graph
        """
        # Extract primitives from each modality
        all_primitives = {}
        for modality, data in inputs.items():
            if modality in self.adapters:
                adapter = self.adapters[modality]
                primitives = adapter.extract_primitives(data)
                
                # Normalize to common representation
                normalized = [adapter.normalize_pattern(p) for p in primitives]
                all_primitives[modality] = normalized
        
        # Find cross-modal correspondences
        correspondences = self._find_correspondences(all_primitives)
        
        # Build unified graph
        unified_graph = self._build_unified_graph(all_primitives, correspondences)
        
        # Discover emergent symbols
        symbols = self._discover_symbols(unified_graph)
        
        # Update symbol grounding
        self._update_symbol_grounding(symbols, all_primitives)
        
        return unified_graph
    
    def _find_correspondences(self, primitives: Dict[str, List[CN]]) -> List[Tuple]:
        """Find patterns that correspond across modalities"""
        correspondences = []
        
        # Compare patterns across modalities
        modalities = list(primitives.keys())
        
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                patterns1 = primitives[mod1]
                patterns2 = primitives[mod2]
                
                # Temporal alignment for time-based modalities
                if self._are_temporal(mod1, mod2):
                    aligned = self._temporal_align(patterns1, patterns2)
                    correspondences.extend(aligned)
                
                # Semantic matching for all modalities
                semantic_matches = self._semantic_match(patterns1, patterns2)
                correspondences.extend(semantic_matches)
        
        return correspondences
    
    def _are_temporal(self, mod1: str, mod2: str) -> bool:
        """Check if modalities have temporal dimension"""
        temporal_mods = {'audio', 'video', 'sensor'}
        return mod1 in temporal_mods or mod2 in temporal_mods
    
    def _temporal_align(self, patterns1: List[CN], patterns2: List[CN]) -> List[Tuple]:
        """Align patterns based on temporal correspondence"""
        aligned = []
        
        for p1 in patterns1:
            # Find temporally closest pattern
            if hasattr(p1, 'position') and len(p1.position) > 0:
                time1 = p1.position[0]
                
                best_match = None
                best_distance = float('inf')
                
                for p2 in patterns2:
                    if hasattr(p2, 'position') and len(p2.position) > 0:
                        time2 = p2.position[0]
                        distance = abs(time1 - time2)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_match = p2
                
                if best_match and best_distance < 0.1:  # 100ms window
                    aligned.append((p1, best_match, 'temporal', 1.0 - best_distance))
        
        return aligned
    
    def _semantic_match(self, patterns1: List[CN], patterns2: List[CN]) -> List[Tuple]:
        """Match patterns based on semantic similarity"""
        matches = []
        
        for p1 in patterns1:
            if not hasattr(p1, 'features'):
                continue
                
            for p2 in patterns2:
                if not hasattr(p2, 'features'):
                    continue
                
                # Cross-modal feature comparison
                similarity = self._cross_modal_similarity(p1, p2)
                
                if similarity > 0.7:  # Threshold
                    matches.append((p1, p2, 'semantic', similarity))
        
        return matches
    
    def _cross_modal_similarity(self, p1: CN, p2: CN) -> float:
        """Compute similarity between patterns from different modalities"""
        # Get normalized features
        features1 = p1.features if hasattr(p1, 'features') else p1.Et
        features2 = p2.features if hasattr(p2, 'features') else p2.Et
        
        # Ensure compatible dimensions
        min_dim = min(len(features1), len(features2))
        f1 = features1[:min_dim]
        f2 = features2[:min_dim]
        
        # Cosine similarity
        similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-7)
        
        # Modality-specific adjustments
        if p1.modality != p2.modality:
            # Cross-modal similarity is typically lower
            similarity *= 0.8
            
            # Boost similarity for known associations
            if (p1.modality, p2.modality) in [('vision', 'text'), ('audio', 'vision')]:
                similarity *= 1.2
        
        return np.clip(similarity, 0, 1)
    
    def _build_unified_graph(self, primitives: Dict[str, List[CN]], 
                           correspondences: List[Tuple]) -> CN:
        """Build unified graph connecting all modalities"""
        unified = CN()
        unified.modality = 'unified'
        unified.N_ = []
        unified.L_ = []
        
        # Add all normalized primitives
        for modality, patterns in primitives.items():
            unified.N_.extend(patterns)
        
        # Create links from correspondences
        for p1, p2, corr_type, strength in correspondences:
            link = CN()
            link.N_ = [p1, p2]
            link.Et = np.array([strength, 0, 1])  # Match, diff, count
            link.correspondence_type = corr_type
            unified.L_.append(link)
        
        # Compute unified features
        unified.Et = np.sum([n.Et for n in unified.N_], axis=0)
        
        return unified
    
    def _discover_symbols(self, graph: CN) -> List[CN]:
        """Discover emergent symbols from cross-modal patterns"""
        symbols = []
        
        # Find highly connected cross-modal nodes
        node_connections = {}
        for link in graph.L_:
            for node in link.N_:
                if node not in node_connections:
                    node_connections[node] = []
                node_connections[node].extend([n for n in link.N_ if n != node])
        
        # Identify symbol candidates
        for node, connected in node_connections.items():
            # Check if connected across multiple modalities
            modalities = set(n.modality for n in connected)
            
            if len(modalities) >= 2:  # Cross-modal connection
                symbol = CN()
                symbol.modality = 'symbol'
                symbol.grounded_nodes = [node] + connected
                symbol.grounded_modalities = list(modalities)
                
                # Symbol strength based on connections
                symbol.Et = np.array([
                    len(connected),  # Connectivity
                    len(modalities),  # Cross-modal span
                    np.mean([n.Et[0] for n in connected])  # Average strength
                ])
                
                symbols.append(symbol)
        
        return symbols
    
    def _update_symbol_grounding(self, symbols: List[CN], 
                               primitives: Dict[str, List[CN]]):
        """Update mapping between abstract symbols and concrete instances"""
        for symbol in symbols:
            # Create symbol ID based on modalities and features
            modality_str = '-'.join(sorted(symbol.grounded_modalities))
            
            # Compute symbol signature
            features = []
            for node in symbol.grounded_nodes:
                if hasattr(node, 'features'):
                    features.append(node.features)
            
            if features:
                signature = np.mean(features, axis=0)
                symbol_id = f"{modality_str}_{hash(signature.tobytes()) % 10000}"
                
                # Store grounding
                if symbol_id not in self.symbol_grounding:
                    self.symbol_grounding[symbol_id] = {
                        'instances': [],
                        'modalities': symbol.grounded_modalities,
                        'signature': signature
                    }
                
                self.symbol_grounding[symbol_id]['instances'].append(symbol)
    
    def query_cross_modal(self, query_input: Any, query_modality: str,
                         target_modality: str) -> List[CN]:
        """
        Query for patterns in target modality given input in query modality
        
        Example: Given an image, find corresponding sounds
        """
        # Process query input
        adapter = self.adapters[query_modality]
        query_primitives = adapter.extract_primitives(query_input)
        query_normalized = [adapter.normalize_pattern(p) for p in query_primitives]
        
        # Find cross-modal matches in stored patterns
        matches = []
        
        for symbol_id, grounding in self.symbol_grounding.items():
            if query_modality in grounding['modalities'] and \
               target_modality in grounding['modalities']:
                
                # Check if query matches this symbol
                for instance in grounding['instances']:
                    for node in instance.grounded_nodes:
                        if node.modality == query_modality:
                            # Compare with query
                            for q_node in query_normalized:
                                similarity = self._cross_modal_similarity(q_node, node)
                                
                                if similarity > 0.8:
                                    # Find corresponding target modality nodes
                                    for target_node in instance.grounded_nodes:
                                        if target_node.modality == target_modality:
                                            matches.append((target_node, similarity))
        
        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return [match[0] for match in matches]


# Example usage and testing

def test_cross_modal_cogalg():
    """Test cross-modal pattern discovery"""
    
    # Initialize system
    cm_cogalg = CrossModalCogAlg()
    
    # Create synthetic multimodal data
    # Vision: Simple geometric pattern
    image = np.zeros((100, 100))
    image[40:60, 40:60] = 255  # White square
    
    # Audio: Corresponding beep
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Text: Description
    text = "square bright object center screen"
    
    # Process multimodal input
    inputs = {
        'vision': image,
        'audio': audio,
        'text': text
    }
    
    print("Processing multimodal input...")
    unified_graph = cm_cogalg.process_multimodal_input(inputs)
    
    print(f"Unified graph contains {len(unified_graph.N_)} nodes")
    print(f"Found {len(unified_graph.L_)} cross-modal links")
    print(f"Discovered {len(cm_cogalg.symbol_grounding)} symbols")
    
    # Test cross-modal query
    print("\nTesting cross-modal query...")
    query_image = np.zeros((100, 100))
    query_image[30:50, 30:50] = 200  # Similar square
    
    matches = cm_cogalg.query_cross_modal(query_image, 'vision', 'audio')
    print(f"Found {len(matches)} audio patterns matching the visual query")
    
    return cm_cogalg


if __name__ == "__main__":
    test_cross_modal_cogalg()
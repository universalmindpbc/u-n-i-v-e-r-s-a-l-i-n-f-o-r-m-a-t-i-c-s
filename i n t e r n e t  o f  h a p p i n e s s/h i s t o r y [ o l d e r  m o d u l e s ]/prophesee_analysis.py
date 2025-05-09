import numpy as np
import cv2
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union, Any
from prophesee_sdk_implementation import PropheseeProcessor

class EventVisualizationType(Enum):
    """Types of event visualizations available."""
    STANDARD = "standard"
    HEATMAP = "heatmap"
    FLOW = "flow"
    CONTOUR = "contour"
    OVERLAY = "overlay"

class EventAnalytics:
    """
    Advanced analytics for Prophesee event-based camera data.
    Provides specialized analysis and visualization capabilities.
    """
    def __init__(self, processor=None):
        """
        Initialize the EventAnalytics module
        
        Args:
            processor: Optional PropheseeProcessor instance (will create one if not provided)
        """
        self.processor = processor or PropheseeProcessor()
        self.event_buffer = []
        self.motion_metrics = {
            "velocity": 0.0,
            "acceleration": 0.0,
            "density": 0.0,
            "spatial_entropy": 0.0,
            "temporal_entropy": 0.0
        }
    
    def calculate_motion_metrics(self, events):
        """
        Calculate advanced motion metrics from event data
        
        Args:
            events: Event data from Prophesee camera
            
        Returns:
            Dictionary of motion metrics
        """
        # Process optical flow
        flow_data = self.processor.process_event_stream(events, output_format='numpy')
        
        # Calculate velocity (magnitude of flow vectors)
        flow_magnitude = np.sqrt(flow_data[:,:,0]**2 + flow_data[:,:,1]**2)
        
        # Event density (normalized count)
        if hasattr(events, 'size'):
            density = min(1.0, events.size / (flow_data.shape[0] * flow_data.shape[1]))
        else:
            density = 0.5  # Default if events don't have size attribute
        
        # Update metrics
        self.motion_metrics.update({
            "velocity": float(np.mean(flow_magnitude)),
            "density": float(density),
            "spatial_entropy": float(self._calculate_spatial_entropy(flow_data)),
        })
        
        return self.motion_metrics.copy()
    
    def _calculate_spatial_entropy(self, flow_data):
        """Calculate spatial entropy of the flow field"""
        # Normalize flow magnitude
        flow_mag = np.sqrt(flow_data[:,:,0]**2 + flow_data[:,:,1]**2)
        flow_mag_norm = flow_mag / (np.max(flow_mag) + 1e-6)
        
        # Create histogram
        hist, _ = np.histogram(flow_mag_norm, bins=32, range=(0, 1), density=True)
        hist = hist / np.sum(hist)
        
        # Remove zeros
        hist = hist[hist > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy / 5.0  # Normalize to 0-1 range (5.0 is approximate max entropy)
    
    def visualize_events(self, events, visualization_type=EventVisualizationType.STANDARD, 
                        output_size=(640, 480), overlay_image=None):
        """
        Create visualization of event data
        
        Args:
            events: Event data
            visualization_type: Type of visualization to create
            output_size: Size of output visualization
            overlay_image: Optional background image for overlay mode
            
        Returns:
            Visualization image as numpy array
        """
        # Process to flow frame
        flow_frame = self.processor.process_event_stream(events)
        
        # Resize if needed
        if flow_frame.shape[0] != output_size[1] or flow_frame.shape[1] != output_size[0]:
            flow_frame = cv2.resize(flow_frame, output_size)
        
        # Create visualization based on type
        if visualization_type == EventVisualizationType.HEATMAP:
            # Create heatmap visualization
            flow_magnitude = np.sqrt(flow_frame[:,:,0]**2 + flow_frame[:,:,1]**2)
            normalized = flow_magnitude / (np.max(flow_magnitude) + 1e-6)
            heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            return heatmap
            
        elif visualization_type == EventVisualizationType.FLOW:
            # Create flow visualization (similar to optical flow)
            hsv = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            hsv[..., 1] = 255
            
            # Calculate magnitude and angle
            mag, ang = cv2.cartToPolar(flow_frame[..., 0], flow_frame[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
        elif visualization_type == EventVisualizationType.OVERLAY and overlay_image is not None:
            # Overlay events on background image
            heatmap = self.visualize_events(events, EventVisualizationType.HEATMAP, output_size)
            
            # Resize overlay image if needed
            if overlay_image.shape[:2] != (output_size[1], output_size[0]):
                overlay_image = cv2.resize(overlay_image, output_size)
                
            # Blend images
            return cv2.addWeighted(overlay_image, 0.7, heatmap, 0.3, 0)
            
        else:  # STANDARD
            # Simple event visualization
            # Convert optical flow to RGB
            return self._flow_to_rgb(flow_frame)
    
    def _flow_to_rgb(self, flow_data):
        """Convert optical flow data to RGB visualization"""
        # Normalize flow components
        fx = flow_data[:,:,0]
        fy = flow_data[:,:,1]
        
        # Normalize
        radius = np.sqrt(fx*fx + fy*fy)
        max_radius = np.max(radius) + 1e-6
        
        # Create RGB image: R channel for x motion, G channel for y motion
        rgb = np.zeros((flow_data.shape[0], flow_data.shape[1], 3), dtype=np.uint8)
        
        # Positive x motion -> red, negative x motion -> cyan
        rgb[:,:,0] = np.where(fx > 0, 255 * np.abs(fx) / max_radius, 0).astype(np.uint8)
        
        # Positive y motion -> green, negative y motion -> magenta
        rgb[:,:,1] = np.where(fy > 0, 255 * np.abs(fy) / max_radius, 0).astype(np.uint8)
        
        # Add blue component for negative motions
        rgb[:,:,2] = np.where((fx < 0) | (fy < 0), 
                            255 * np.maximum(np.abs(fx), np.abs(fy)) / max_radius, 
                            0).astype(np.uint8)
        
        return rgb

# Create global instance for easy import
event_analytics = EventAnalytics() 
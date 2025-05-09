#!/usr/bin/env python3
"""
Prophesee SDK Integration Demo
Demonstrates the functionality of the Prophesee SDK in the Universal Light Algorithm framework
"""

import numpy as np
import cv2
import time
import os
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any

# Import Prophesee components
from prophesee_sdk_implementation import PropheseeProcessor
from prophesee_analysis import EventAnalytics, EventVisualizationType
from prophesee_device_integration import DeviceIntegration, InputSource

def main():
    """Main demo function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prophesee SDK Integration Demo')
    parser.add_argument('--mode', type=str, default='simulation', 
                       choices=['simulation', 'file', 'camera', 'iphone', 'voyage81'],
                       help='Demo mode')
    parser.add_argument('--input', type=str, default=None,
                       help='Input file path for file mode')
    parser.add_argument('--device-id', type=int, default=0,
                       help='Device ID for camera modes')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for saved visualizations')
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    processor = PropheseeProcessor()
    analytics = EventAnalytics(processor)
    device = DeviceIntegration(processor, analytics)
    
    print("Prophesee SDK Integration Demo")
    print("-----------------------------")
    
    # Set up device based on mode
    if args.mode == 'simulation':
        print("Running in simulation mode...")
        # Create simulated event data
        width, height = 640, 480
        event_count = 5000
        
        # Create synthetic events in a circular pattern
        events = create_synthetic_events(width, height, event_count)
        
        # Demo processing
        run_processing_demo(events, processor, analytics, args.output_dir)
        
    elif args.mode == 'file':
        if not args.input:
            print("Error: Input file path required for file mode")
            return
            
        print(f"Loading events from file: {args.input}")
        
        # Connect to file
        if device.connect_device(InputSource.RECORDED, args.input):
            # Capture events
            events = device.capture_events()
            
            # Process and visualize
            run_processing_demo(events, processor, analytics, args.output_dir)
            
            # Disconnect
            device.disconnect()
        else:
            print("Failed to load event data file")
    
    elif args.mode in ('camera', 'iphone', 'voyage81'):
        # Map mode to source type
        source_map = {
            'camera': InputSource.PROPHESEE,
            'iphone': InputSource.IPHONE,
            'voyage81': InputSource.VOYAGE81
        }
        
        source_type = source_map[args.mode]
        print(f"Connecting to {args.mode} device (ID: {args.device_id})...")
        
        # Connect to device
        if device.connect_device(source_type, args.device_id):
            print("Connected successfully")
            
            try:
                # Continuous capture and processing demo
                run_live_demo(device, analytics, args.output_dir)
            except KeyboardInterrupt:
                print("\nDemo stopped by user")
            finally:
                # Disconnect
                device.disconnect()
                print("Device disconnected")
        else:
            print(f"Failed to connect to {args.mode} device")

def create_synthetic_events(width, height, event_count):
    """
    Create synthetic events in a circular pattern
    
    Args:
        width: Frame width
        height: Frame height
        event_count: Number of events to generate
        
    Returns:
        Synthetic events array
    """
    # Generate timestamps (microseconds, increasing)
    t = np.linspace(0, 1e6, event_count)
    
    # Generate coordinates in a circular pattern
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4
    
    angle = np.linspace(0, 4 * np.pi, event_count)
    x = center_x + radius * np.cos(angle)
    y = center_y + radius * np.sin(angle)
    
    # Add some noise
    x += np.random.normal(0, 5, event_count)
    y += np.random.normal(0, 5, event_count)
    
    # Ensure within frame bounds
    x = np.clip(x, 0, width - 1).astype(int)
    y = np.clip(y, 0, height - 1).astype(int)
    
    # Random polarities
    p = np.random.choice([-1, 1], event_count)
    
    # Create events array
    events = np.column_stack((x, y, p, t))
    
    return events

def run_processing_demo(events, processor, analytics, output_dir):
    """
    Run a processing demo on a batch of events
    
    Args:
        events: Event data
        processor: PropheseeProcessor instance
        analytics: EventAnalytics instance
        output_dir: Output directory for visualizations
    """
    print(f"Processing {len(events)} events...")
    
    # Initialize pipeline if not already
    if processor.trail_filter is None:
        processor.initialize_pipeline(640, 480)
    
    # Calculate motion metrics
    metrics = analytics.calculate_motion_metrics(events)
    print("Motion Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Create different visualizations
    print("Generating visualizations...")
    
    # Standard visualization
    standard_vis = analytics.visualize_events(events)
    cv2.imwrite(os.path.join(output_dir, 'standard_visualization.png'), standard_vis)
    
    # Heatmap visualization
    heatmap_vis = analytics.visualize_events(events, EventVisualizationType.HEATMAP)
    cv2.imwrite(os.path.join(output_dir, 'heatmap_visualization.png'), heatmap_vis)
    
    # Flow visualization
    flow_vis = analytics.visualize_events(events, EventVisualizationType.FLOW)
    cv2.imwrite(os.path.join(output_dir, 'flow_visualization.png'), flow_vis)
    
    print(f"Visualizations saved to {output_dir}")

def run_live_demo(device, analytics, output_dir):
    """
    Run a live demo with continuous capture and processing
    
    Args:
        device: DeviceIntegration instance
        analytics: EventAnalytics instance
        output_dir: Output directory for visualizations
    """
    print("Starting live demo... Press Ctrl+C to stop")
    
    # Create window for visualization
    cv2.namedWindow('Event Visualization', cv2.WINDOW_NORMAL)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        # Capture events
        events = device.capture_events(duration_ms=50)
        
        if events is None or len(events) == 0:
            print("No events captured")
            time.sleep(0.1)
            continue
        
        # Process events
        metrics = analytics.calculate_motion_metrics(events)
        
        # Create visualization
        visualization = analytics.visualize_events(events, EventVisualizationType.FLOW)
        
        # Add metrics text
        cv2.putText(visualization, f"Events: {len(events)}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(visualization, f"Velocity: {metrics['velocity']:.2f}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(visualization, f"Density: {metrics['density']:.2f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display visualization
        cv2.imshow('Event Visualization', visualization)
        
        # Save every 100th frame
        frame_count += 1
        if frame_count % 100 == 0:
            cv2.imwrite(os.path.join(output_dir, f'live_frame_{frame_count}.png'), visualization)
        
        # Calculate FPS
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = 10 / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.1f}, Events: {len(events)}")
            start_time = time.time()
        
        # Check for exit
        if cv2.waitKey(1) == 27:  # ESC key
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
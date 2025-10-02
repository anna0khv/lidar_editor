"""
3D Visualization Module
Provides interactive 3D visualization of point clouds with editing capabilities
"""

import open3d as o3d
import numpy as np
import logging
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
import threading
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SelectionRegion:
    """Represents a selected region in 3D space"""
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    indices: np.ndarray
    region_type: str  # "box", "sphere", "polygon"


class InteractiveVisualizer:
    """Interactive 3D visualizer for point cloud editing"""
    
    def __init__(self):
        self.vis = None
        self.point_cloud = None
        self.original_colors = None
        self.selected_indices = set()
        self.selection_regions = []
        self.callbacks = {}
        self.is_running = False
        
        # Visualization settings
        self.colors = {
            'original': [0.7, 0.7, 0.7],      # Gray
            'selected': [1.0, 0.0, 0.0],      # Red
            'dynamic': [1.0, 0.5, 0.0],       # Orange
            'static': [0.0, 1.0, 0.0],        # Green
            'ground': [0.5, 0.3, 0.1],        # Brown
            'uncertain': [1.0, 1.0, 0.0],     # Yellow
        }
        
        # Selection tools
        self.selection_mode = "box"  # "box", "sphere", "lasso"
        self.selection_size = 2.0
    
    def initialize(self, window_name: str = "LIDAR Editor", width: int = 1200, height: int = 800):
        """Initialize the visualizer window"""
        try:
            self.vis = o3d.visualization.VisualizerWithEditing()
            self.vis.create_window(window_name, width, height)
            
            # Set up rendering options
            render_option = self.vis.get_render_option()
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            render_option.point_size = 2.0
            render_option.show_coordinate_frame = True
            
            logger.info("Visualizer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize visualizer: {e}")
            return False
    
    def set_point_cloud(self, point_cloud: o3d.geometry.PointCloud):
        """Set the point cloud to visualize"""
        self.point_cloud = point_cloud.copy()
        
        # Store original colors or create default ones
        if len(point_cloud.colors) > 0:
            self.original_colors = np.asarray(point_cloud.colors).copy()
        else:
            self.original_colors = np.tile(self.colors['original'], (len(point_cloud.points), 1))
            self.point_cloud.colors = o3d.utility.Vector3dVector(self.original_colors)
        
        # Clear previous selections
        self.selected_indices.clear()
        self.selection_regions.clear()
        
        if self.vis is not None:
            self.vis.clear_geometries()
            self.vis.add_geometry(self.point_cloud)
            self.vis.reset_view_point(True)
    
    def color_points_by_classification(self, dynamic_indices: np.ndarray, static_indices: np.ndarray, 
                                     ground_indices: Optional[np.ndarray] = None):
        """Color points based on classification results"""
        if self.point_cloud is None:
            return
        
        colors = self.original_colors.copy()
        
        # Color dynamic objects
        if len(dynamic_indices) > 0:
            colors[dynamic_indices] = self.colors['dynamic']
        
        # Color static objects
        if len(static_indices) > 0:
            colors[static_indices] = self.colors['static']
        
        # Color ground points
        if ground_indices is not None and len(ground_indices) > 0:
            colors[ground_indices] = self.colors['ground']
        
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        if self.vis is not None:
            self.vis.update_geometry(self.point_cloud)
    
    def highlight_selection(self, indices: np.ndarray):
        """Highlight selected points"""
        if self.point_cloud is None:
            return
        
        colors = np.asarray(self.point_cloud.colors)
        
        # Reset previous selection colors
        for idx in self.selected_indices:
            if idx < len(colors):
                colors[idx] = self.original_colors[idx]
        
        # Highlight new selection
        self.selected_indices = set(indices)
        colors[indices] = self.colors['selected']
        
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        if self.vis is not None:
            self.vis.update_geometry(self.point_cloud)
    
    def select_box_region(self, center: np.ndarray, size: np.ndarray) -> np.ndarray:
        """Select points within a box region"""
        if self.point_cloud is None:
            return np.array([])
        
        points = np.asarray(self.point_cloud.points)
        
        # Create box bounds
        min_bound = center - size / 2
        max_bound = center + size / 2
        
        # Find points within bounds
        within_bounds = np.all(
            (points >= min_bound) & (points <= max_bound), axis=1
        )
        
        indices = np.where(within_bounds)[0]
        
        # Create selection region
        region = SelectionRegion(
            center=center,
            size=size,
            rotation=np.eye(3),
            indices=indices,
            region_type="box"
        )
        self.selection_regions.append(region)
        
        return indices
    
    def select_sphere_region(self, center: np.ndarray, radius: float) -> np.ndarray:
        """Select points within a spherical region"""
        if self.point_cloud is None:
            return np.array([])
        
        points = np.asarray(self.point_cloud.points)
        
        # Calculate distances from center
        distances = np.linalg.norm(points - center, axis=1)
        
        # Find points within radius
        within_radius = distances <= radius
        indices = np.where(within_radius)[0]
        
        # Create selection region
        region = SelectionRegion(
            center=center,
            size=np.array([radius, radius, radius]),
            rotation=np.eye(3),
            indices=indices,
            region_type="sphere"
        )
        self.selection_regions.append(region)
        
        return indices
    
    def delete_selected_points(self) -> int:
        """Delete currently selected points"""
        if self.point_cloud is None or len(self.selected_indices) == 0:
            return 0
        
        # Get points and colors to keep
        all_indices = np.arange(len(self.point_cloud.points))
        keep_mask = ~np.isin(all_indices, list(self.selected_indices))
        
        points = np.asarray(self.point_cloud.points)[keep_mask]
        colors = np.asarray(self.point_cloud.colors)[keep_mask]
        
        # Update point cloud
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # Update original colors
        self.original_colors = self.original_colors[keep_mask]
        
        # Clear selection
        deleted_count = len(self.selected_indices)
        self.selected_indices.clear()
        
        if self.vis is not None:
            self.vis.update_geometry(self.point_cloud)
        
        logger.info(f"Deleted {deleted_count} points")
        return deleted_count
    
    def copy_selected_points(self) -> Optional[np.ndarray]:
        """Copy selected points to clipboard (return as array)"""
        if self.point_cloud is None or len(self.selected_indices) == 0:
            return None
        
        points = np.asarray(self.point_cloud.points)
        selected_points = points[list(self.selected_indices)]
        
        logger.info(f"Copied {len(selected_points)} points")
        return selected_points
    
    def paste_points(self, points: np.ndarray, offset: np.ndarray = np.array([0, 0, 0])):
        """Paste points at specified offset"""
        if self.point_cloud is None or len(points) == 0:
            return
        
        # Apply offset
        new_points = points + offset
        
        # Combine with existing points
        existing_points = np.asarray(self.point_cloud.points)
        combined_points = np.vstack([existing_points, new_points])
        
        # Create colors for new points
        existing_colors = np.asarray(self.point_cloud.colors)
        new_colors = np.tile(self.colors['original'], (len(new_points), 1))
        combined_colors = np.vstack([existing_colors, new_colors])
        
        # Update point cloud
        self.point_cloud.points = o3d.utility.Vector3dVector(combined_points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(combined_colors)
        
        # Update original colors
        self.original_colors = np.vstack([self.original_colors, new_colors])
        
        if self.vis is not None:
            self.vis.update_geometry(self.point_cloud)
        
        logger.info(f"Pasted {len(new_points)} points")
    
    def add_callback(self, event: str, callback: Callable):
        """Add callback for events"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def trigger_callback(self, event: str, *args, **kwargs):
        """Trigger callbacks for an event"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in callback for {event}: {e}")
    
    def run(self):
        """Run the visualizer (blocking)"""
        if self.vis is None:
            logger.error("Visualizer not initialized")
            return
        
        self.is_running = True
        
        try:
            # Add key callbacks
            self.vis.register_key_callback(ord('D'), self._on_delete_key)
            self.vis.register_key_callback(ord('C'), self._on_copy_key)
            self.vis.register_key_callback(ord('V'), self._on_paste_key)
            self.vis.register_key_callback(ord('S'), self._on_save_key)
            
            logger.info("Starting visualizer. Press 'D' to delete, 'C' to copy, 'V' to paste, 'S' to save")
            self.vis.run()
            
        except Exception as e:
            logger.error(f"Error running visualizer: {e}")
        finally:
            self.is_running = False
            self.vis.destroy_window()
    
    def _on_delete_key(self, vis):
        """Handle delete key press"""
        if len(self.selected_indices) > 0:
            self.delete_selected_points()
            self.trigger_callback('points_deleted', list(self.selected_indices))
        return False
    
    def _on_copy_key(self, vis):
        """Handle copy key press"""
        copied_points = self.copy_selected_points()
        if copied_points is not None:
            self.trigger_callback('points_copied', copied_points)
        return False
    
    def _on_paste_key(self, vis):
        """Handle paste key press"""
        # This would need to be implemented with a clipboard system
        self.trigger_callback('paste_requested')
        return False
    
    def _on_save_key(self, vis):
        """Handle save key press"""
        self.trigger_callback('save_requested')
        return False
    
    def close(self):
        """Close the visualizer"""
        if self.vis is not None and self.is_running:
            self.vis.close()
    
    def get_current_point_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        """Get the current point cloud"""
        return self.point_cloud
    
    def reset_colors(self):
        """Reset all points to original colors"""
        if self.point_cloud is not None:
            self.point_cloud.colors = o3d.utility.Vector3dVector(self.original_colors)
            if self.vis is not None:
                self.vis.update_geometry(self.point_cloud)
    
    def get_selection_stats(self) -> Dict:
        """Get statistics about current selection"""
        if len(self.selected_indices) == 0:
            return {}
        
        points = np.asarray(self.point_cloud.points)
        selected_points = points[list(self.selected_indices)]
        
        return {
            'count': len(self.selected_indices),
            'center': selected_points.mean(axis=0),
            'bounds': {
                'min': selected_points.min(axis=0),
                'max': selected_points.max(axis=0)
            }
        }


def test_visualizer():
    """Test the visualizer"""
    from point_cloud_loader import PointCloudLoader
    
    loader = PointCloudLoader()
    if loader.load_pcd("data/points.pcd"):
        vis = InteractiveVisualizer()
        if vis.initialize():
            vis.set_point_cloud(loader.get_point_cloud())
            
            # Add some test callbacks
            def on_save():
                print("Save requested!")
            
            vis.add_callback('save_requested', on_save)
            
            # Run visualizer
            vis.run()
    else:
        print("Failed to load test file")


if __name__ == "__main__":
    test_visualizer()


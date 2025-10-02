"""
Point Cloud Loader Module
Handles loading and saving of PCD files using Open3D
"""

import open3d as o3d
import numpy as np
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PointCloudLoader:
    """Class for loading and saving point cloud data"""
    
    def __init__(self):
        self.point_cloud = None
        # self.original_points = None
        self.metadata = {}
    
    def load_pcd(self, file_path: str, downsample_for_preview: bool = True, max_points_preview: int = 1000000) -> bool:
        """
        Load PCD file with optional downsampling for large files
        
        Args:
            file_path: Path to PCD file
            downsample_for_preview: Whether to downsample large files for faster loading
            max_points_preview: Maximum points to keep for preview (if downsampling)
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False

            logger.info(f"Loading PCD file: {file_path}")
            t0 = time.perf_counter()
            
            # Load the point cloud
            pc = o3d.io.read_point_cloud(str(file_path))
            t1 = time.perf_counter()
            logger.info(f"Initial load: {t1-t0:.3f}s")

            if len(pc.points) == 0:
                logger.error("No points found in the file")
                return False

            original_count = len(pc.points)
            logger.info(f"Original point count: {original_count:,}")
            
            # Downsample if file is too large
            if downsample_for_preview and original_count > max_points_preview:
                logger.info(f"Downsampling from {original_count:,} to ~{max_points_preview:,} points for faster visualization")
                
                # Calculate voxel size for downsampling
                # Use approximate voxel size to get desired number of points
                points = np.asarray(pc.points)
                bbox_size = points.max(axis=0) - points.min(axis=0)
                total_volume = np.prod(bbox_size)
                target_density = max_points_preview / total_volume
                voxel_size = (1.0 / target_density) ** (1/3)
                
                # Apply voxel downsampling
                t2 = time.perf_counter()
                pc_downsampled = pc.voxel_down_sample(voxel_size)
                t3 = time.perf_counter()
                
                logger.info(f"Downsampling: {t3-t2:.3f}s, reduced to {len(pc_downsampled.points):,} points")
                self.point_cloud = pc_downsampled
                self.original_points = pc  # Store original for later use
                
                # Mark as downsampled in metadata
                self.metadata = {
                    'num_points': len(self.point_cloud.points),
                    'original_num_points': original_count,
                    'is_downsampled': True,
                    'downsample_ratio': len(self.point_cloud.points) / original_count,
                    'has_colors': len(self.point_cloud.colors) > 0,
                    'has_normals': len(self.point_cloud.normals) > 0,
                    'file_path': str(file_path),
                    'bounds': None  # compute on demand
                }
            else:
                self.point_cloud = pc
                self.original_points = None
                
                self.metadata = {
                    'num_points': len(self.point_cloud.points),
                    'original_num_points': original_count,
                    'is_downsampled': False,
                    'downsample_ratio': 1.0,
                    'has_colors': len(self.point_cloud.colors) > 0,
                    'has_normals': len(self.point_cloud.normals) > 0,
                    'file_path': str(file_path),
                    'bounds': None  # compute on demand
                }

            total_time = time.perf_counter() - t0
            logger.info(f"Successfully loaded in {total_time:.3f}s - showing {self.metadata['num_points']:,} points")
            return True

        except Exception as e:
            logger.error(f"Error loading PCD file: {e}")
            return False

    
    def save_pcd(self, file_path: str, point_cloud: Optional[o3d.geometry.PointCloud] = None) -> bool:
        """
        Save point cloud to PCD file
        
        Args:
            file_path: Output file path
            point_cloud: Point cloud to save (uses self.point_cloud if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            pc_to_save = point_cloud if point_cloud is not None else self.point_cloud
            
            if pc_to_save is None:
                logger.error("No point cloud to save")
                return False
            
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving PCD file: {file_path}")
            success = o3d.io.write_point_cloud(str(file_path), pc_to_save)
            
            if success:
                logger.info(f"Successfully saved {len(pc_to_save.points)} points")
            else:
                logger.error("Failed to save PCD file")
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving PCD file: {e}")
            return False
    
    def get_bounds(self) -> Dict[str, Tuple[float, float, float]]:
        """Get bounding box of the point cloud"""
        if self.point_cloud is None:
            return {}
        
        points = np.asarray(self.point_cloud.points)
        if len(points) == 0:
            return {}
        
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center = (min_bound + max_bound) / 2
        
        return {
            'min': tuple(min_bound),
            'max': tuple(max_bound),
            'center': tuple(center),
            'size': tuple(max_bound - min_bound)
        }
    
    def get_point_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        """Get the current point cloud"""
        return self.point_cloud
    
    def get_points_array(self) -> Optional[np.ndarray]:
        """Get points as numpy array"""
        if self.point_cloud is None:
            return None
        return np.asarray(self.point_cloud.points)
    
    def get_colors_array(self) -> Optional[np.ndarray]:
        """Get colors as numpy array"""
        if self.point_cloud is None or len(self.point_cloud.colors) == 0:
            return None
        return np.asarray(self.point_cloud.colors)
    
    def set_points(self, points: np.ndarray):
        """Set new points for the point cloud"""
        if self.point_cloud is None:
            self.point_cloud = o3d.geometry.PointCloud()
        
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        
        # Update metadata
        self.metadata['num_points'] = len(points)
        self.metadata['bounds'] = self.get_bounds()
    
    def add_colors(self, colors: np.ndarray):
        """Add colors to the point cloud"""
        if self.point_cloud is None:
            logger.error("No point cloud loaded")
            return
        
        if len(colors) != len(self.point_cloud.points):
            logger.error("Color array size doesn't match points array size")
            return
        
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        self.metadata['has_colors'] = True
    
    def remove_points(self, indices: np.ndarray):
        """Remove points at specified indices"""
        if self.point_cloud is None:
            logger.error("No point cloud loaded")
            return
        
        # Create mask for points to keep
        all_indices = np.arange(len(self.point_cloud.points))
        keep_mask = ~np.isin(all_indices, indices)
        
        # Get points and colors to keep
        points = np.asarray(self.point_cloud.points)[keep_mask]
        
        # Update point cloud
        self.point_cloud.points = o3d.utility.Vector3dVector(points)
        
        # Update colors if they exist
        if len(self.point_cloud.colors) > 0:
            colors = np.asarray(self.point_cloud.colors)[keep_mask]
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # Update metadata
        self.metadata['num_points'] = len(points)
        self.metadata['bounds'] = self.get_bounds()
        
        logger.info(f"Removed {len(indices)} points, {len(points)} points remaining")
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the loaded point cloud"""
        return self.metadata.copy()
    
    def get_original_point_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        """Get the original (non-downsampled) point cloud if available"""
        return self.original_points
    
    def is_downsampled(self) -> bool:
        """Check if current point cloud is downsampled"""
        return self.metadata.get('is_downsampled', False)
    
    def get_downsample_info(self) -> Dict[str, Any]:
        """Get downsampling information"""
        if not self.is_downsampled():
            return {}
        
        return {
            'is_downsampled': True,
            'original_count': self.metadata.get('original_num_points', 0),
            'current_count': self.metadata.get('num_points', 0),
            'ratio': self.metadata.get('downsample_ratio', 1.0),
            'reduction_percent': (1 - self.metadata.get('downsample_ratio', 1.0)) * 100
        }


def test_loader():
    """Test the point cloud loader"""
    loader = PointCloudLoader()
    
    # Test loading
    test_file = "data/points.pcd"
    if loader.load_pcd(test_file):
        info = loader.get_info()
        print(f"Loaded point cloud info: {info}")
        
        # Test saving
        output_file = "data/test_output.pcd"
        if loader.save_pcd(output_file):
            print(f"Successfully saved to {output_file}")
    else:
        print("Failed to load test file")


if __name__ == "__main__":
    test_loader()


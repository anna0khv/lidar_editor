"""
Dynamic Object Detection Module
Implements algorithms to automatically detect and remove dynamic objects from point clouds
"""

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of dynamic object detection"""
    dynamic_indices: np.ndarray  # Indices of points classified as dynamic
    static_indices: np.ndarray   # Indices of points classified as static
    clusters: List[np.ndarray]   # List of point clusters
    confidence_scores: np.ndarray  # Confidence score for each point
    method_used: str             # Detection method used


class DynamicObjectDetector:
    """Class for detecting dynamic objects in point clouds"""
    
    def __init__(self):
        self.point_cloud = None
        self.ground_plane = None
        self.detection_params = {
            'height_threshold': 0.2,      # Points above ground plane
            'cluster_eps': 0.5,           # DBSCAN epsilon
            'cluster_min_samples': 10,    # DBSCAN min samples
            'vehicle_height_range': (0.5, 3.0),  # Typical vehicle height
            'vehicle_width_range': (1.0, 3.0),   # Typical vehicle width
            'vehicle_length_range': (2.0, 8.0),  # Typical vehicle length
            'density_threshold': 0.1,     # Point density threshold
            'road_width_estimate': 20.0,  # Estimated road width for filtering
        }
    
    def set_point_cloud(self, point_cloud: o3d.geometry.PointCloud):
        """Set the point cloud for processing"""
        self.point_cloud = point_cloud
        self.ground_plane = None
    
    def detect_ground_plane(self) -> Optional[np.ndarray]:
        """
        Detect ground plane using RANSAC
        
        Returns:
            Ground plane coefficients [a, b, c, d] for ax + by + cz + d = 0
        """
        if self.point_cloud is None:
            logger.error("No point cloud set")
            return None
        
        try:
            # Use RANSAC to find ground plane
            plane_model, inliers = self.point_cloud.segment_plane(
                distance_threshold=0.1,
                ransac_n=3,
                num_iterations=1000
            )
            
            self.ground_plane = plane_model
            logger.info(f"Ground plane detected: {plane_model}")
            logger.info(f"Ground plane inliers: {len(inliers)}")
            
            return plane_model
            
        except Exception as e:
            logger.error(f"Error detecting ground plane: {e}")
            return None
    
    def filter_by_height(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter points by height above ground plane
        
        Args:
            points: Point array (N, 3)
            
        Returns:
            Tuple of (above_ground_indices, ground_indices)
        """
        if self.ground_plane is None:
            logger.warning("No ground plane detected, using Z-coordinate")
            # Use simple Z-threshold if no ground plane
            z_coords = points[:, 2]
            z_threshold = np.percentile(z_coords, 10) + self.detection_params['height_threshold']
            above_ground = z_coords > z_threshold
            return np.where(above_ground)[0], np.where(~above_ground)[0]
        
        # Calculate distance from each point to ground plane
        a, b, c, d = self.ground_plane
        distances = np.abs(
            a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
        ) / np.sqrt(a**2 + b**2 + c**2)
        
        above_ground = distances > self.detection_params['height_threshold']
        return np.where(above_ground)[0], np.where(~above_ground)[0]
    
    def cluster_points(self, points: np.ndarray, indices: np.ndarray) -> List[np.ndarray]:
        """
        Cluster points using optimized DBSCAN for large datasets
        
        Args:
            points: Point array
            indices: Indices of points to cluster
            
        Returns:
            List of cluster indices
        """
        if len(indices) == 0:
            return []
        
        cluster_points = points[indices]
        
        logger.info(f"Clustering {len(cluster_points)} points...")
        
        # For large datasets, use optimized approach
        if len(cluster_points) > 30000:
            logger.info("Large dataset detected, using optimized clustering...")
            
            # Sample points for faster clustering
            sample_ratio = min(0.3, 20000 / len(cluster_points))
            sample_size = int(len(cluster_points) * sample_ratio)
            sample_indices = np.random.choice(len(cluster_points), sample_size, replace=False)
            sample_points = cluster_points[sample_indices]
            
            logger.info(f"Clustering on {len(sample_points)} sampled points")
            
            # Apply DBSCAN clustering on sample
            clustering = DBSCAN(
                eps=self.detection_params['cluster_eps'],
                min_samples=max(5, self.detection_params['cluster_min_samples'] // 2)
            ).fit(sample_points)
            
            # Find cluster centers and assign all points
            clusters = []
            unique_labels = np.unique(clustering.labels_)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                # Get cluster center
                cluster_mask = clustering.labels_ == label
                cluster_sample_points = sample_points[cluster_mask]
                cluster_center = np.mean(cluster_sample_points, axis=0)
                
                # Find all points within cluster radius of center
                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                cluster_radius = self.detection_params['cluster_eps'] * 1.5
                within_cluster = distances <= cluster_radius
                
                if np.sum(within_cluster) >= self.detection_params['cluster_min_samples']:
                    cluster_indices = indices[within_cluster]
                    clusters.append(cluster_indices)
            
        else:
            # Standard DBSCAN for smaller datasets
            clustering = DBSCAN(
                eps=self.detection_params['cluster_eps'],
                min_samples=self.detection_params['cluster_min_samples']
            ).fit(cluster_points)
            
            clusters = []
            unique_labels = np.unique(clustering.labels_)
            
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                
                cluster_mask = clustering.labels_ == label
                cluster_indices = indices[cluster_mask]
                clusters.append(cluster_indices)
        
        logger.info(f"Found {len(clusters)} clusters from {len(indices)} points")
        return clusters
    
    def analyze_cluster_geometry(self, points: np.ndarray, cluster_indices: np.ndarray) -> Dict:
        """
        Analyze geometric properties of a cluster
        
        Args:
            points: All points
            cluster_indices: Indices of points in the cluster
            
        Returns:
            Dictionary with geometric properties
        """
        cluster_points = points[cluster_indices]
        
        # Calculate bounding box
        min_bound = cluster_points.min(axis=0)
        max_bound = cluster_points.max(axis=0)
        dimensions = max_bound - min_bound
        
        # Calculate center and volume
        center = (min_bound + max_bound) / 2
        volume = np.prod(dimensions)
        
        # Calculate point density
        density = len(cluster_points) / max(volume, 1e-6)
        
        # Check if dimensions match typical vehicles
        height, width, length = sorted(dimensions)
        
        vehicle_like = (
            self.detection_params['vehicle_height_range'][0] <= height <= self.detection_params['vehicle_height_range'][1] and
            self.detection_params['vehicle_width_range'][0] <= width <= self.detection_params['vehicle_width_range'][1] and
            self.detection_params['vehicle_length_range'][0] <= length <= self.detection_params['vehicle_length_range'][1]
        )
        
        return {
            'center': center,
            'dimensions': dimensions,
            'volume': volume,
            'density': density,
            'num_points': len(cluster_points),
            'vehicle_like': vehicle_like,
            'min_bound': min_bound,
            'max_bound': max_bound
        }
    
    def classify_clusters(self, points: np.ndarray, clusters: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Classify clusters as dynamic or static based on geometric properties
        
        Args:
            points: All points
            clusters: List of cluster indices
            
        Returns:
            Tuple of (dynamic_clusters, static_clusters)
        """
        dynamic_clusters = []
        static_clusters = []
        
        for cluster_indices in clusters:
            if len(cluster_indices) < 5:  # Too few points
                static_clusters.append(cluster_indices)
                continue
            
            geometry = self.analyze_cluster_geometry(points, cluster_indices)
            
            # Classification logic
            is_dynamic = False
            
            # Check if it looks like a vehicle
            if geometry['vehicle_like']:
                is_dynamic = True
                logger.info(f"Vehicle-like cluster detected: {geometry['dimensions']}")
            
            # Check density (vehicles typically have lower density than buildings)
            elif geometry['density'] < self.detection_params['density_threshold']:
                is_dynamic = True
                logger.info(f"Low-density cluster detected: density={geometry['density']:.4f}")
            
            # Check if it's too high (unlikely to be ground infrastructure)
            elif geometry['dimensions'][2] > 4.0:  # Very tall objects
                is_dynamic = True
                logger.info(f"Tall object detected: height={geometry['dimensions'][2]:.2f}")
            
            if is_dynamic:
                dynamic_clusters.append(cluster_indices)
            else:
                static_clusters.append(cluster_indices)
        
        logger.info(f"Classified {len(dynamic_clusters)} dynamic and {len(static_clusters)} static clusters")
        return dynamic_clusters, static_clusters
    
    def detect_dynamic_objects(self, method: str = "geometric") -> DetectionResult:
        """
        Main method to detect dynamic objects
        
        Args:
            method: Detection method ("geometric", "statistical", "hybrid")
            
        Returns:
            DetectionResult object
        """
        if self.point_cloud is None:
            raise ValueError("No point cloud set")
        
        points = np.asarray(self.point_cloud.points)
        
        logger.info(f"Starting dynamic object detection with method: {method}")
        logger.info(f"Processing {len(points)} points")
        
        # Step 1: Detect ground plane
        self.detect_ground_plane()
        
        # Step 2: Filter points by height
        above_ground_indices, ground_indices = self.filter_by_height(points)
        logger.info(f"Found {len(above_ground_indices)} points above ground")
        
        # Step 3: Cluster above-ground points
        clusters = self.cluster_points(points, above_ground_indices)
        
        # Step 4: Classify clusters
        dynamic_clusters, static_clusters = self.classify_clusters(points, clusters)
        
        # Step 5: Combine results
        dynamic_indices = np.concatenate(dynamic_clusters) if dynamic_clusters else np.array([], dtype=int)
        static_indices = np.concatenate([ground_indices] + static_clusters) if static_clusters else ground_indices
        
        # Step 6: Calculate confidence scores (simple version)
        confidence_scores = np.ones(len(points))
        confidence_scores[dynamic_indices] = 0.8  # High confidence for dynamic
        confidence_scores[static_indices] = 0.9   # High confidence for static
        
        result = DetectionResult(
            dynamic_indices=dynamic_indices,
            static_indices=static_indices,
            clusters=clusters,
            confidence_scores=confidence_scores,
            method_used=method
        )
        
        logger.info(f"Detection complete: {len(dynamic_indices)} dynamic points, {len(static_indices)} static points")
        
        return result
    
    def refine_detection(self, result: DetectionResult, user_corrections: Dict) -> DetectionResult:
        """
        Refine detection results based on user corrections
        
        Args:
            result: Initial detection result
            user_corrections: Dictionary with user corrections
            
        Returns:
            Refined DetectionResult
        """
        # This method would implement machine learning refinement
        # based on user feedback - placeholder for now
        logger.info("Refining detection based on user feedback")
        return result


def test_detector():
    """Test the dynamic object detector"""
    from point_cloud_loader import PointCloudLoader
    
    loader = PointCloudLoader()
    if loader.load_pcd("data/points.pcd"):
        detector = DynamicObjectDetector()
        detector.set_point_cloud(loader.get_point_cloud())
        
        result = detector.detect_dynamic_objects()
        print(f"Detection result: {len(result.dynamic_indices)} dynamic points")
        print(f"Found {len(result.clusters)} clusters")
    else:
        print("Failed to load test file")


if __name__ == "__main__":
    test_detector()


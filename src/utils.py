"""
Utility functions for point cloud processing
"""

import numpy as np
import open3d as o3d
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


def downsample_point_cloud(point_cloud: o3d.geometry.PointCloud, voxel_size: float = 0.1) -> o3d.geometry.PointCloud:
    """
    Downsample point cloud using voxel grid
    
    Args:
        point_cloud: Input point cloud
        voxel_size: Size of voxel grid
        
    Returns:
        Downsampled point cloud
    """
    return point_cloud.voxel_down_sample(voxel_size)


def remove_outliers(point_cloud: o3d.geometry.PointCloud, 
                   nb_neighbors: int = 20, 
                   std_ratio: float = 2.0) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Remove statistical outliers from point cloud
    
    Args:
        point_cloud: Input point cloud
        nb_neighbors: Number of neighbors to consider
        std_ratio: Standard deviation ratio threshold
        
    Returns:
        Tuple of (cleaned point cloud, inlier indices)
    """
    cleaned_pc, inlier_indices = point_cloud.remove_statistical_outlier(nb_neighbors, std_ratio)
    return cleaned_pc, np.array(inlier_indices)


def estimate_normals(point_cloud: o3d.geometry.PointCloud, 
                    radius: float = 0.5, 
                    max_nn: int = 30) -> o3d.geometry.PointCloud:
    """
    Estimate normals for point cloud
    
    Args:
        point_cloud: Input point cloud
        radius: Search radius for normal estimation
        max_nn: Maximum number of neighbors
        
    Returns:
        Point cloud with estimated normals
    """
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return point_cloud


def segment_plane_ransac(point_cloud: o3d.geometry.PointCloud,
                        distance_threshold: float = 0.1,
                        ransac_n: int = 3,
                        num_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment plane using RANSAC
    
    Args:
        point_cloud: Input point cloud
        distance_threshold: Distance threshold for plane fitting
        ransac_n: Number of points to sample for RANSAC
        num_iterations: Number of RANSAC iterations
        
    Returns:
        Tuple of (plane coefficients, inlier indices)
    """
    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    return np.array(plane_model), np.array(inliers)


def compute_point_density(points: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """
    Compute local point density for each point
    
    Args:
        points: Point array (N, 3)
        radius: Search radius for density computation
        
    Returns:
        Density values for each point
    """
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(radius=radius).fit(points)
    distances, indices = nbrs.radius_neighbors(points)
    
    densities = np.array([len(idx) for idx in indices])
    return densities


def filter_by_height_range(point_cloud: o3d.geometry.PointCloud,
                          min_height: float,
                          max_height: float,
                          ground_plane: Optional[np.ndarray] = None) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Filter points by height range
    
    Args:
        point_cloud: Input point cloud
        min_height: Minimum height
        max_height: Maximum height
        ground_plane: Ground plane coefficients [a, b, c, d] (optional)
        
    Returns:
        Tuple of (filtered point cloud, kept indices)
    """
    points = np.asarray(point_cloud.points)
    
    if ground_plane is not None:
        # Calculate height relative to ground plane
        a, b, c, d = ground_plane
        heights = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
    else:
        # Use Z coordinate as height
        heights = points[:, 2]
    
    # Filter by height range
    height_mask = (heights >= min_height) & (heights <= max_height)
    kept_indices = np.where(height_mask)[0]
    
    # Create filtered point cloud
    filtered_pc = point_cloud.select_by_index(kept_indices)
    
    return filtered_pc, kept_indices


def compute_bounding_boxes(point_cloud: o3d.geometry.PointCloud, 
                          cluster_indices: List[np.ndarray]) -> List[Dict]:
    """
    Compute bounding boxes for point clusters
    
    Args:
        point_cloud: Input point cloud
        cluster_indices: List of arrays containing indices for each cluster
        
    Returns:
        List of bounding box information dictionaries
    """
    points = np.asarray(point_cloud.points)
    bounding_boxes = []
    
    for cluster_idx in cluster_indices:
        if len(cluster_idx) == 0:
            continue
            
        cluster_points = points[cluster_idx]
        
        # Compute axis-aligned bounding box
        min_bound = cluster_points.min(axis=0)
        max_bound = cluster_points.max(axis=0)
        center = (min_bound + max_bound) / 2
        size = max_bound - min_bound
        volume = np.prod(size)
        
        # Compute oriented bounding box
        cluster_pc = o3d.geometry.PointCloud()
        cluster_pc.points = o3d.utility.Vector3dVector(cluster_points)
        obb = cluster_pc.get_oriented_bounding_box()
        
        bbox_info = {
            'cluster_indices': cluster_idx,
            'num_points': len(cluster_idx),
            'center': center,
            'size': size,
            'volume': volume,
            'min_bound': min_bound,
            'max_bound': max_bound,
            'oriented_bbox': obb
        }
        
        bounding_boxes.append(bbox_info)
    
    return bounding_boxes


def merge_close_clusters(clusters: List[np.ndarray], 
                        points: np.ndarray, 
                        distance_threshold: float = 2.0) -> List[np.ndarray]:
    """
    Merge clusters that are close to each other
    
    Args:
        clusters: List of cluster indices
        points: Point array
        distance_threshold: Maximum distance between cluster centers to merge
        
    Returns:
        List of merged cluster indices
    """
    if len(clusters) <= 1:
        return clusters
    
    # Compute cluster centers
    centers = []
    for cluster_idx in clusters:
        if len(cluster_idx) > 0:
            center = points[cluster_idx].mean(axis=0)
            centers.append(center)
        else:
            centers.append(np.array([0, 0, 0]))
    
    centers = np.array(centers)
    
    # Find clusters to merge
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(centers))
    
    merged_clusters = []
    used_indices = set()
    
    for i, cluster_i in enumerate(clusters):
        if i in used_indices:
            continue
            
        # Find all clusters close to this one
        close_indices = np.where(distances[i] < distance_threshold)[0]
        
        # Merge all close clusters
        merged_cluster = cluster_i.copy()
        for j in close_indices:
            if j != i and j not in used_indices:
                merged_cluster = np.concatenate([merged_cluster, clusters[j]])
                used_indices.add(j)
        
        merged_clusters.append(merged_cluster)
        used_indices.add(i)
    
    logger.info(f"Merged {len(clusters)} clusters into {len(merged_clusters)} clusters")
    return merged_clusters


def create_mesh_from_points(point_cloud: o3d.geometry.PointCloud, 
                           method: str = "poisson") -> o3d.geometry.TriangleMesh:
    """
    Create mesh from point cloud
    
    Args:
        point_cloud: Input point cloud with normals
        method: Meshing method ("poisson" or "ball_pivoting")
        
    Returns:
        Generated triangle mesh
    """
    if method == "poisson":
        # Poisson surface reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=9
        )
    elif method == "ball_pivoting":
        # Ball pivoting algorithm
        radii = [0.005, 0.01, 0.02, 0.04]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            point_cloud, o3d.utility.DoubleVector(radii)
        )
    else:
        raise ValueError(f"Unknown meshing method: {method}")
    
    return mesh


def analyze_point_distribution(points: np.ndarray) -> Dict:
    """
    Analyze the distribution of points in 3D space
    
    Args:
        points: Point array (N, 3)
        
    Returns:
        Dictionary with distribution statistics
    """
    stats = {
        'num_points': len(points),
        'mean': points.mean(axis=0),
        'std': points.std(axis=0),
        'min': points.min(axis=0),
        'max': points.max(axis=0),
        'range': points.max(axis=0) - points.min(axis=0),
        'centroid': points.mean(axis=0)
    }
    
    # Compute covariance matrix and eigenvalues
    cov_matrix = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    stats['covariance_matrix'] = cov_matrix
    stats['eigenvalues'] = eigenvalues
    stats['eigenvectors'] = eigenvectors
    stats['principal_directions'] = eigenvectors
    
    # Compute point density statistics
    if len(points) > 1:
        from scipy.spatial.distance import pdist
        distances = pdist(points)
        stats['mean_distance'] = distances.mean()
        stats['std_distance'] = distances.std()
        stats['min_distance'] = distances.min()
        stats['max_distance'] = distances.max()
    
    return stats


def export_detection_results(detection_result, output_path: str):
    """
    Export detection results to file
    
    Args:
        detection_result: DetectionResult object
        output_path: Output file path
    """
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    export_data = {
        'dynamic_indices': detection_result.dynamic_indices.tolist(),
        'static_indices': detection_result.static_indices.tolist(),
        'num_clusters': len(detection_result.clusters),
        'method_used': detection_result.method_used,
        'confidence_scores_stats': {
            'mean': float(detection_result.confidence_scores.mean()),
            'std': float(detection_result.confidence_scores.std()),
            'min': float(detection_result.confidence_scores.min()),
            'max': float(detection_result.confidence_scores.max())
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Detection results exported to {output_path}")


def load_detection_results(input_path: str) -> Dict:
    """
    Load detection results from file
    
    Args:
        input_path: Input file path
        
    Returns:
        Dictionary with detection results
    """
    import json
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays
    data['dynamic_indices'] = np.array(data['dynamic_indices'])
    data['static_indices'] = np.array(data['static_indices'])
    
    logger.info(f"Detection results loaded from {input_path}")
    return data

#!/usr/bin/env python3
"""
Test script for LIDAR Editor components
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from point_cloud_loader import PointCloudLoader
from dynamic_object_detector import DynamicObjectDetector
from utils import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_point_cloud_loader():
    """Test point cloud loading functionality"""
    print("\n=== Testing Point Cloud Loader ===")
    
    loader = PointCloudLoader()
    
    # Test loading original file
    original_file = "data/points.pcd"
    if Path(original_file).exists():
        print(f"Loading {original_file}...")
        success = loader.load_pcd(original_file)
        
        if success:
            info = loader.get_info()
            print(f"✓ Successfully loaded {info['num_points']:,} points")
            print(f"  File: {info['file_path']}")
            print(f"  Bounds: {info['bounds']}")
            print(f"  Has colors: {info['has_colors']}")
            print(f"  Has normals: {info['has_normals']}")
            
            # Test saving
            test_output = "data/test_loader_output.pcd"
            if loader.save_pcd(test_output):
                print(f"✓ Successfully saved to {test_output}")
            else:
                print("✗ Failed to save file")
                
        else:
            print("✗ Failed to load file")
    else:
        print(f"✗ File not found: {original_file}")
    
    # Test loading processed file
    processed_file = "data/processed_points.pcd"
    if Path(processed_file).exists():
        print(f"\nLoading {processed_file}...")
        success = loader.load_pcd(processed_file)
        
        if success:
            info = loader.get_info()
            print(f"✓ Successfully loaded {info['num_points']:,} points")
        else:
            print("✗ Failed to load processed file")
    else:
        print(f"✗ File not found: {processed_file}")


def test_dynamic_object_detector():
    """Test dynamic object detection"""
    print("\n=== Testing Dynamic Object Detector ===")
    
    loader = PointCloudLoader()
    detector = DynamicObjectDetector()
    
    # Load test data
    test_file = "data/points.pcd"
    if not Path(test_file).exists():
        print(f"✗ Test file not found: {test_file}")
        return
    
    print(f"Loading {test_file}...")
    if not loader.load_pcd(test_file):
        print("✗ Failed to load test file")
        return
    
    point_cloud = loader.get_point_cloud()
    detector.set_point_cloud(point_cloud)
    
    print("Running detection...")
    try:
        result = detector.detect_dynamic_objects("geometric")
        
        print(f"✓ Detection completed successfully")
        print(f"  Dynamic points: {len(result.dynamic_indices):,}")
        print(f"  Static points: {len(result.static_indices):,}")
        print(f"  Clusters found: {len(result.clusters)}")
        print(f"  Method used: {result.method_used}")
        
        # Analyze clusters
        if result.clusters:
            cluster_sizes = [len(cluster) for cluster in result.clusters]
            print(f"  Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")
        
        # Save detection results
        from utils import export_detection_results
        export_detection_results(result, "data/test_detection_results.json")
        print("✓ Detection results saved")
        
        return result
        
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        return None


def test_utilities():
    """Test utility functions"""
    print("\n=== Testing Utility Functions ===")
    
    loader = PointCloudLoader()
    
    # Load test data
    test_file = "data/points.pcd"
    if not Path(test_file).exists():
        print(f"✗ Test file not found: {test_file}")
        return
    
    if not loader.load_pcd(test_file):
        print("✗ Failed to load test file")
        return
    
    point_cloud = loader.get_point_cloud()
    original_size = len(point_cloud.points)
    print(f"Original point cloud size: {original_size:,}")
    
    # Test downsampling
    print("\nTesting downsampling...")
    downsampled = downsample_point_cloud(point_cloud, voxel_size=0.2)
    print(f"✓ Downsampled to {len(downsampled.points):,} points (reduction: {(1 - len(downsampled.points)/original_size)*100:.1f}%)")
    
    # Test outlier removal
    print("\nTesting outlier removal...")
    try:
        cleaned, inliers = remove_outliers(point_cloud, nb_neighbors=20, std_ratio=2.0)
        print(f"✓ Removed {original_size - len(cleaned.points):,} outliers")
    except Exception as e:
        print(f"✗ Outlier removal failed: {e}")
    
    # Test normal estimation
    print("\nTesting normal estimation...")
    try:
        pc_with_normals = estimate_normals(point_cloud.copy(), radius=0.5)
        print(f"✓ Estimated normals for {len(pc_with_normals.normals)} points")
    except Exception as e:
        print(f"✗ Normal estimation failed: {e}")
    
    # Test plane segmentation
    print("\nTesting plane segmentation...")
    try:
        plane_model, inliers = segment_plane_ransac(point_cloud)
        print(f"✓ Found plane with {len(inliers):,} inliers")
        print(f"  Plane equation: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")
    except Exception as e:
        print(f"✗ Plane segmentation failed: {e}")
    
    # Test point distribution analysis
    print("\nTesting point distribution analysis...")
    try:
        points = np.asarray(point_cloud.points)
        stats = analyze_point_distribution(points)
        print(f"✓ Point distribution analysis completed")
        print(f"  Mean: ({stats['mean'][0]:.2f}, {stats['mean'][1]:.2f}, {stats['mean'][2]:.2f})")
        print(f"  Range: ({stats['range'][0]:.2f}, {stats['range'][1]:.2f}, {stats['range'][2]:.2f})")
    except Exception as e:
        print(f"✗ Point distribution analysis failed: {e}")


def test_performance():
    """Test performance with different file sizes"""
    print("\n=== Testing Performance ===")
    
    loader = PointCloudLoader()
    
    test_files = ["data/points.pcd", "data/processed_points.pcd"]
    
    for test_file in test_files:
        if not Path(test_file).exists():
            continue
            
        print(f"\nTesting with {test_file}...")
        
        # Time loading
        import time
        start_time = time.time()
        success = loader.load_pcd(test_file)
        load_time = time.time() - start_time
        
        if success:
            info = loader.get_info()
            print(f"✓ Loaded {info['num_points']:,} points in {load_time:.2f} seconds")
            print(f"  Loading rate: {info['num_points']/load_time:.0f} points/second")
            
            # Time detection
            detector = DynamicObjectDetector()
            detector.set_point_cloud(loader.get_point_cloud())
            
            start_time = time.time()
            try:
                result = detector.detect_dynamic_objects("geometric")
                detection_time = time.time() - start_time
                print(f"✓ Detection completed in {detection_time:.2f} seconds")
                print(f"  Detection rate: {info['num_points']/detection_time:.0f} points/second")
            except Exception as e:
                print(f"✗ Detection failed: {e}")
        else:
            print(f"✗ Failed to load {test_file}")


def main():
    """Run all tests"""
    print("LIDAR Editor Component Tests")
    print("=" * 50)
    
    # Run tests
    test_point_cloud_loader()
    test_dynamic_object_detector()
    test_utilities()
    test_performance()
    
    print("\n" + "=" * 50)
    print("Testing completed!")


if __name__ == "__main__":
    main()

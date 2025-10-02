#!/usr/bin/env python3
"""
Simple test without Open3D to check basic functionality
"""

import sys
import numpy as np
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

def test_basic_imports():
    """Test basic imports without Open3D"""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    return True

def test_pcd_file_reading():
    """Test basic PCD file reading without Open3D"""
    print("\nTesting PCD file structure...")
    
    pcd_file = "data/points.pcd"
    if not Path(pcd_file).exists():
        print(f"✗ PCD file not found: {pcd_file}")
        return False
    
    try:
        with open(pcd_file, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header_info = {}
        data_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('VERSION'):
                header_info['version'] = line.split()[1]
            elif line.startswith('FIELDS'):
                header_info['fields'] = line.split()[1:]
            elif line.startswith('SIZE'):
                header_info['sizes'] = [int(x) for x in line.split()[1:]]
            elif line.startswith('TYPE'):
                header_info['types'] = line.split()[1:]
            elif line.startswith('COUNT'):
                header_info['counts'] = [int(x) for x in line.split()[1:]]
            elif line.startswith('WIDTH'):
                header_info['width'] = int(line.split()[1])
            elif line.startswith('HEIGHT'):
                header_info['height'] = int(line.split()[1])
            elif line.startswith('POINTS'):
                header_info['points'] = int(line.split()[1])
            elif line.startswith('DATA'):
                header_info['data_format'] = line.split()[1]
                data_start = i + 1
                break
        
        print(f"✓ PCD header parsed successfully")
        print(f"  Version: {header_info.get('version', 'unknown')}")
        print(f"  Fields: {header_info.get('fields', [])}")
        print(f"  Points: {header_info.get('points', 0):,}")
        print(f"  Data format: {header_info.get('data_format', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ PCD file reading failed: {e}")
        return False

def test_clustering_algorithm():
    """Test clustering algorithm with synthetic data"""
    print("\nTesting clustering algorithm...")
    
    try:
        from sklearn.cluster import DBSCAN
        
        # Generate synthetic 3D point cloud data
        np.random.seed(42)
        
        # Create clusters
        cluster1 = np.random.normal([0, 0, 0], 0.5, (100, 3))
        cluster2 = np.random.normal([5, 5, 0], 0.5, (80, 3))
        cluster3 = np.random.normal([10, 0, 2], 0.3, (60, 3))
        noise = np.random.uniform([-2, -2, -1], [12, 7, 3], (50, 3))
        
        # Combine all points
        points = np.vstack([cluster1, cluster2, cluster3, noise])
        
        print(f"  Generated {len(points)} synthetic points")
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=1.0, min_samples=5).fit(points)
        
        # Analyze results
        labels = clustering.labels_
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"✓ DBSCAN clustering completed")
        print(f"  Found {n_clusters} clusters")
        print(f"  Noise points: {n_noise}")
        
        return True
        
    except Exception as e:
        print(f"✗ Clustering test failed: {e}")
        return False

def test_geometric_analysis():
    """Test geometric analysis functions"""
    print("\nTesting geometric analysis...")
    
    try:
        # Generate test data
        np.random.seed(42)
        points = np.random.uniform([0, 0, 0], [10, 10, 5], (1000, 3))
        
        # Test bounding box calculation
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center = (min_bound + max_bound) / 2
        size = max_bound - min_bound
        
        print(f"✓ Bounding box calculated")
        print(f"  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"  Size: ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})")
        
        # Test height filtering
        ground_height = min_bound[2] + 0.5
        above_ground = points[points[:, 2] > ground_height]
        
        print(f"✓ Height filtering completed")
        print(f"  Ground level: {ground_height:.2f}")
        print(f"  Points above ground: {len(above_ground)}")
        
        # Test density calculation
        from scipy.spatial.distance import pdist
        distances = pdist(points[:100])  # Sample for performance
        mean_distance = distances.mean()
        
        print(f"✓ Density analysis completed")
        print(f"  Mean distance: {mean_distance:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Geometric analysis failed: {e}")
        return False

def test_vehicle_detection_logic():
    """Test vehicle detection logic"""
    print("\nTesting vehicle detection logic...")
    
    try:
        # Define vehicle parameters
        vehicle_params = {
            'height_range': (0.5, 3.0),
            'width_range': (1.0, 3.0),
            'length_range': (2.0, 8.0),
            'min_points': 20
        }
        
        # Test cases
        test_objects = [
            {'dimensions': [2.5, 1.8, 1.5], 'points': 150, 'expected': True, 'name': 'Car'},
            {'dimensions': [8.0, 2.5, 3.0], 'points': 300, 'expected': True, 'name': 'Truck'},
            {'dimensions': [0.5, 0.5, 1.8], 'points': 25, 'expected': False, 'name': 'Person'},
            {'dimensions': [10.0, 10.0, 15.0], 'points': 1000, 'expected': False, 'name': 'Building'},
            {'dimensions': [1.5, 1.5, 0.3], 'points': 15, 'expected': False, 'name': 'Small object'}
        ]
        
        correct_classifications = 0
        
        for obj in test_objects:
            dims = obj['dimensions']
            points = obj['points']
            
            # Sort dimensions to get height, width, length
            sorted_dims = sorted(dims)
            height, width, length = sorted_dims
            
            # Check if it matches vehicle criteria
            is_vehicle = (
                vehicle_params['height_range'][0] <= height <= vehicle_params['height_range'][1] and
                vehicle_params['width_range'][0] <= width <= vehicle_params['width_range'][1] and
                vehicle_params['length_range'][0] <= length <= vehicle_params['length_range'][1] and
                points >= vehicle_params['min_points']
            )
            
            if is_vehicle == obj['expected']:
                correct_classifications += 1
                result = "✓"
            else:
                result = "✗"
            
            print(f"  {result} {obj['name']}: {dims} -> {'Vehicle' if is_vehicle else 'Not vehicle'}")
        
        accuracy = correct_classifications / len(test_objects)
        print(f"✓ Vehicle detection logic test completed")
        print(f"  Accuracy: {accuracy:.1%} ({correct_classifications}/{len(test_objects)})")
        
        return accuracy >= 0.8
        
    except Exception as e:
        print(f"✗ Vehicle detection logic failed: {e}")
        return False

def main():
    """Run all simple tests"""
    print("LIDAR Editor Simple Tests")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_pcd_file_reading,
        test_clustering_algorithm,
        test_geometric_analysis,
        test_vehicle_detection_logic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Tests completed: {passed}/{total} passed ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")

if __name__ == "__main__":
    main()

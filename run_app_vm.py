#!/usr/bin/env python3
"""
VM-compatible version of LIDAR Editor
Runs without Open3D for virtual machines
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

def check_vm_compatibility():
    """Check if we're running on a VM and what's available"""
    print("🔍 Проверка совместимости с виртуальной машиной...")
    
    # Check available libraries
    available_libs = {}
    
    try:
        import numpy
        available_libs['numpy'] = numpy.__version__
        print(f"✅ NumPy {numpy.__version__}")
    except ImportError:
        print("❌ NumPy не найден")
        return False
    
    try:
        import sklearn
        available_libs['sklearn'] = sklearn.__version__
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError:
        print("❌ Scikit-learn не найден")
        return False
    
    try:
        import matplotlib
        available_libs['matplotlib'] = matplotlib.__version__
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib не найден")
    
    # Check Open3D (likely to fail on VM)
    try:
        import open3d
        available_libs['open3d'] = open3d.__version__
        print(f"✅ Open3D {open3d.__version__}")
    except ImportError:
        print("⚠️ Open3D не найден (нормально для VM)")
    except Exception as e:
        print(f"⚠️ Open3D недоступен: {e}")
    
    # Check PyQt6 (may fail on VM without display)
    try:
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 доступен")
        available_libs['pyqt6'] = True
    except ImportError:
        print("⚠️ PyQt6 не найден")
    except Exception as e:
        print(f"⚠️ PyQt6 недоступен: {e}")
    
    return available_libs

def run_vm_demo():
    """Run VM-compatible demo"""
    print("\n🎯 Запуск демонстрации для виртуальной машины")
    print("=" * 50)
    
    # Import our basic modules
    try:
        import numpy as np
        from sklearn.cluster import DBSCAN
        
        print("\n🤖 Тестирование алгоритма обнаружения...")
        
        # Generate synthetic data
        np.random.seed(42)
        
        # Create test point cloud
        ground = np.random.uniform([0, 0, 0], [50, 50, 0.2], (1000, 3))
        cars = np.random.uniform([10, 10, 0.5], [15, 12, 2.0], (100, 3))
        buildings = np.random.uniform([20, 20, 0], [30, 30, 10], (500, 3))
        
        all_points = np.vstack([ground, cars, buildings])
        print(f"✅ Создано {len(all_points)} тестовых точек")
        
        # Filter by height
        above_ground = all_points[all_points[:, 2] > 0.5]
        print(f"✅ Найдено {len(above_ground)} точек выше земли")
        
        # Clustering
        clustering = DBSCAN(eps=2.0, min_samples=10).fit(above_ground)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"✅ Найдено {n_clusters} кластеров")
        
        # Analyze clusters
        vehicle_count = 0
        building_count = 0
        
        for label in set(labels):
            if label == -1:
                continue
                
            cluster_points = above_ground[labels == label]
            if len(cluster_points) < 20:
                continue
                
            # Calculate dimensions
            min_bound = cluster_points.min(axis=0)
            max_bound = cluster_points.max(axis=0)
            dims = max_bound - min_bound
            
            # Simple classification
            if 0.5 <= dims[2] <= 3.0 and 1.0 <= dims[0] <= 8.0 and 1.0 <= dims[1] <= 4.0:
                vehicle_count += 1
                print(f"🚗 Обнаружен автомобиль: {dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f}м")
            else:
                building_count += 1
                print(f"🏢 Обнаружено здание: {dims[0]:.1f}×{dims[1]:.1f}×{dims[2]:.1f}м")
        
        print(f"\n📊 Результаты:")
        print(f"   Автомобилей: {vehicle_count}")
        print(f"   Зданий: {building_count}")
        print(f"   Точность: {'✅ Отлично' if vehicle_count > 0 else '⚠️ Требует настройки'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка демонстрации: {e}")
        return False

def run_text_interface():
    """Run text-based interface for VM"""
    print("\n📱 Текстовый интерфейс для виртуальной машины")
    print("=" * 50)
    
    while True:
        print("\nВыберите действие:")
        print("1. 🧪 Запустить тест алгоритма")
        print("2. 📊 Анализ PCD файлов")
        print("3. ⚙️ Проверка системы")
        print("4. 📚 Показать документацию")
        print("0. 🚪 Выход")
        
        try:
            choice = input("\nВаш выбор (0-4): ").strip()
            
            if choice == "0":
                print("👋 До свидания!")
                break
            elif choice == "1":
                run_vm_demo()
            elif choice == "2":
                analyze_pcd_files()
            elif choice == "3":
                check_vm_compatibility()
            elif choice == "4":
                show_documentation()
            else:
                print("❌ Неверный выбор. Попробуйте снова.")
                
        except KeyboardInterrupt:
            print("\n👋 Программа прервана пользователем")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")

def analyze_pcd_files():
    """Analyze available PCD files"""
    print("\n📊 Анализ PCD файлов")
    print("-" * 30)
    
    pcd_files = [
        "data/points.pcd",
        "data/processed_points.pcd"
    ]
    
    for file_path in pcd_files:
        if not Path(file_path).exists():
            print(f"❌ Файл не найден: {file_path}")
            continue
            
        try:
            # Get file size
            size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            print(f"\n📁 {file_path}")
            print(f"   Размер: {size_mb:.1f} МБ")
            
            # Try to read header
            with open(file_path, 'rb') as f:
                header_lines = []
                for _ in range(15):
                    line = f.readline()
                    try:
                        decoded = line.decode('utf-8').strip()
                        header_lines.append(decoded)
                        if decoded.startswith('DATA'):
                            break
                    except UnicodeDecodeError:
                        break
                
                # Parse info
                for line in header_lines:
                    if line.startswith('POINTS'):
                        points = int(line.split()[1])
                        print(f"   Точек: {points:,}")
                    elif line.startswith('FIELDS'):
                        fields = ' '.join(line.split()[1:])
                        print(f"   Поля: {fields}")
                    elif line.startswith('DATA'):
                        fmt = line.split()[1]
                        print(f"   Формат: {fmt}")
                        
        except Exception as e:
            print(f"   ⚠️ Ошибка чтения: {e}")

def show_documentation():
    """Show available documentation"""
    print("\n📚 Документация проекта")
    print("-" * 30)
    
    docs = [
        ("README.md", "Обзор проекта"),
        ("docs/USAGE.md", "Руководство пользователя"),
        ("docs/ARCHITECTURE.md", "Архитектура системы"),
        ("PROJECT_SUMMARY.md", "Отчет о проекте")
    ]
    
    for doc_file, description in docs:
        if Path(doc_file).exists():
            print(f"✅ {doc_file} - {description}")
        else:
            print(f"❌ {doc_file} - не найден")
    
    print(f"\n💡 Для чтения документации используйте:")
    print(f"   cat README.md")
    print(f"   less docs/USAGE.md")

def main():
    """Main function for VM-compatible version"""
    print("🖥️ LIDAR Editor - Версия для виртуальной машины")
    print("=" * 60)
    
    # Check compatibility first
    available = check_vm_compatibility()
    
    if not available:
        print("\n❌ Критические зависимости не найдены!")
        print("Установите: pip install numpy scikit-learn matplotlib")
        return
    
    print(f"\n✅ Базовые компоненты доступны")
    print(f"⚠️ GUI и 3D визуализация недоступны на VM")
    print(f"🎯 Запускаем текстовый режим...")
    
    # Run text interface
    run_text_interface()

if __name__ == "__main__":
    main()

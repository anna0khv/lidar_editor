#!/usr/bin/env python3
"""
Демонстрационный скрипт для презентации возможностей Редактора лидарных карт
"""

import sys
import os
import time
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_step(step, description):
    """Print formatted step"""
    print(f"\n🔸 Шаг {step}: {description}")
    print("-" * 50)

def demo_file_info():
    """Demonstrate file information extraction"""
    print_header("ДЕМОНСТРАЦИЯ: Анализ PCD файлов")
    
    files_to_analyze = [
        ("data/points.pcd", "Исходная карта с динамическими объектами"),
        ("data/processed_points.pcd", "Базовая обработка (только дороги)")
    ]
    
    for file_path, description in files_to_analyze:
        if not Path(file_path).exists():
            print(f"❌ Файл не найден: {file_path}")
            continue
            
        print(f"\n📁 {description}")
        print(f"   Файл: {file_path}")
        
        # Get file size
        file_size = Path(file_path).stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"   Размер: {size_mb:.1f} МБ")
        
        # Try to read header
        try:
            with open(file_path, 'rb') as f:
                # Read first few lines as text
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
                
                # Parse header info
                for line in header_lines:
                    if line.startswith('POINTS'):
                        points_count = int(line.split()[1])
                        print(f"   Точек: {points_count:,}")
                    elif line.startswith('FIELDS'):
                        fields = ' '.join(line.split()[1:])
                        print(f"   Поля: {fields}")
                    elif line.startswith('DATA'):
                        data_format = line.split()[1]
                        print(f"   Формат: {data_format}")
                        
        except Exception as e:
            print(f"   ⚠️ Ошибка чтения заголовка: {e}")

def demo_algorithm_logic():
    """Demonstrate algorithm logic with synthetic data"""
    print_header("ДЕМОНСТРАЦИЯ: Алгоритм обнаружения")
    
    import numpy as np
    from sklearn.cluster import DBSCAN
    
    print_step(1, "Генерация синтетических данных")
    
    # Generate synthetic point cloud
    np.random.seed(42)
    
    # Ground plane
    ground_points = np.random.uniform([0, 0, 0], [50, 50, 0.2], (5000, 3))
    print(f"   🌍 Точки земли: {len(ground_points):,}")
    
    # Buildings (static objects)
    building1 = np.random.uniform([10, 10, 0], [20, 20, 15], (2000, 3))
    building2 = np.random.uniform([30, 30, 0], [40, 40, 12], (1500, 3))
    static_points = np.vstack([building1, building2])
    print(f"   🏢 Статические объекты: {len(static_points):,}")
    
    # Vehicles (dynamic objects)
    car1 = np.random.uniform([5, 25, 0.2], [9, 27, 1.8], (150, 3))  # Car
    truck1 = np.random.uniform([15, 35, 0.2], [23, 38, 3.2], (300, 3))  # Truck
    car2 = np.random.uniform([25, 15, 0.2], [29, 17, 1.6], (120, 3))  # Car
    dynamic_points = np.vstack([car1, truck1, car2])
    print(f"   🚗 Динамические объекты: {len(dynamic_points):,}")
    
    # Combine all points
    all_points = np.vstack([ground_points, static_points, dynamic_points])
    total_points = len(all_points)
    print(f"   📊 Всего точек: {total_points:,}")
    
    print_step(2, "Фильтрация по высоте")
    
    # Filter by height (above ground)
    height_threshold = 0.5
    above_ground = all_points[all_points[:, 2] > height_threshold]
    print(f"   📏 Порог высоты: {height_threshold} м")
    print(f"   ⬆️ Точек выше земли: {len(above_ground):,}")
    
    print_step(3, "Кластеризация (DBSCAN)")
    
    # Apply DBSCAN clustering
    eps = 2.0
    min_samples = 10
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(above_ground)
    
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"   🎯 Параметры: eps={eps}, min_samples={min_samples}")
    print(f"   🔍 Найдено кластеров: {n_clusters}")
    print(f"   🔸 Шумовых точек: {n_noise}")
    
    print_step(4, "Анализ геометрии кластеров")
    
    vehicle_criteria = {
        'height_range': (0.5, 3.5),
        'width_range': (1.0, 4.0),
        'length_range': (2.0, 10.0),
        'min_points': 50
    }
    
    detected_vehicles = 0
    detected_buildings = 0
    
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
            
        cluster_mask = labels == label
        cluster_points = above_ground[cluster_mask]
        
        if len(cluster_points) < 10:
            continue
            
        # Calculate bounding box
        min_bound = cluster_points.min(axis=0)
        max_bound = cluster_points.max(axis=0)
        dimensions = max_bound - min_bound
        
        # Sort dimensions to get height, width, length
        sorted_dims = sorted(dimensions)
        height, width, length = sorted_dims
        
        # Check vehicle criteria
        is_vehicle = (
            vehicle_criteria['height_range'][0] <= height <= vehicle_criteria['height_range'][1] and
            vehicle_criteria['width_range'][0] <= width <= vehicle_criteria['width_range'][1] and
            vehicle_criteria['length_range'][0] <= length <= vehicle_criteria['length_range'][1] and
            len(cluster_points) >= vehicle_criteria['min_points']
        )
        
        if is_vehicle:
            detected_vehicles += 1
            print(f"   🚗 Транспорт: {len(cluster_points)} точек, размеры: {dimensions[0]:.1f}×{dimensions[1]:.1f}×{dimensions[2]:.1f}")
        else:
            detected_buildings += 1
            print(f"   🏢 Здание: {len(cluster_points)} точек, размеры: {dimensions[0]:.1f}×{dimensions[1]:.1f}×{dimensions[2]:.1f}")
    
    print_step(5, "Результаты классификации")
    
    print(f"   ✅ Обнаружено транспортных средств: {detected_vehicles}")
    print(f"   ✅ Обнаружено зданий: {detected_buildings}")
    print(f"   📊 Точность обнаружения ТС: {detected_vehicles/3*100:.0f}% (ожидалось 3)")

def demo_performance_metrics():
    """Demonstrate performance metrics"""
    print_header("ДЕМОНСТРАЦИЯ: Метрики производительности")
    
    # Simulated performance data
    test_cases = [
        {"points": 1000000, "time_min": 10, "time_max": 30},
        {"points": 5000000, "time_min": 60, "time_max": 180},
        {"points": 10000000, "time_min": 300, "time_max": 900},
    ]
    
    print("📈 Ожидаемая производительность:")
    print(f"{'Размер файла':<15} {'Время обработки':<20} {'Скорость':<20}")
    print("-" * 60)
    
    for case in test_cases:
        points = case["points"]
        time_min = case["time_min"]
        time_max = case["time_max"]
        
        speed_min = points // time_max
        speed_max = points // time_min
        
        points_str = f"{points//1000000}M точек" if points >= 1000000 else f"{points//1000}K точек"
        time_str = f"{time_min//60}-{time_max//60} мин" if time_min >= 60 else f"{time_min}-{time_max} сек"
        speed_str = f"{speed_min//1000}-{speed_max//1000}K т/с"
        
        print(f"{points_str:<15} {time_str:<20} {speed_str:<20}")

def demo_features():
    """Demonstrate key features"""
    print_header("ДЕМОНСТРАЦИЯ: Ключевые возможности")
    
    features = [
        ("🤖 Автоматическое обнаружение", [
            "RANSAC для обнаружения плоскости земли",
            "DBSCAN кластеризация объектов",
            "Геометрический анализ кластеров",
            "Классификация по критериям ТС"
        ]),
        ("🎨 Интерактивное редактирование", [
            "3D визуализация с Open3D",
            "Инструменты выделения областей",
            "Операции: удаление, копирование, вставка",
            "Цветовая индикация результатов"
        ]),
        ("💾 Работа с данными", [
            "Поддержка PCD формата (ASCII/бинарный)",
            "Обработка файлов до нескольких ГБ",
            "Экспорт результатов обнаружения",
            "Сохранение в стандартном PCD формате"
        ]),
        ("🚀 Производительность", [
            "Многопоточная обработка",
            "Оптимизированные алгоритмы",
            "Эффективное использование памяти",
            "Прогресс-индикаторы для длительных операций"
        ])
    ]
    
    for feature_name, feature_list in features:
        print(f"\n{feature_name}")
        for item in feature_list:
            print(f"   • {item}")

def main():
    """Main demo function"""
    print("🎯 ДЕМОНСТРАЦИЯ РЕДАКТОРА ЛИДАРНЫХ КАРТ")
    print("Интеллектуальный инструмент для очистки лидарных карт")
    
    try:
        demo_file_info()
        time.sleep(1)
        
        demo_algorithm_logic()
        time.sleep(1)
        
        demo_performance_metrics()
        time.sleep(1)
        
        demo_features()
        
        print_header("ЗАКЛЮЧЕНИЕ")
        print("""
🎉 Редактор лидарных карт предоставляет:

✅ Автоматическое обнаружение динамических объектов
✅ Интерактивные инструменты ручного редактирования  
✅ Высокую производительность обработки
✅ Удобный графический интерфейс
✅ Поддержку стандартных форматов данных

🚀 Готов к использованию для очистки лидарных карт!

Запуск приложения: python run_app.py
Тестирование: python simple_test.py
        """)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Демонстрация прервана пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка демонстрации: {e}")

if __name__ == "__main__":
    main()

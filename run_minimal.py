#!/usr/bin/env python3
"""
Minimal version for problematic VMs
Uses only standard Python libraries
"""

import sys
import os
import struct
import math
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50)

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 6):
        print("❌ Требуется Python 3.6+")
        return False
    
    print("✅ Версия Python подходит")
    return True

def test_basic_libraries():
    """Test basic Python libraries"""
    print("\n🔍 Проверка базовых библиотек:")
    
    libraries = {
        'math': 'Математические функции',
        'struct': 'Работа с бинарными данными', 
        'pathlib': 'Работа с путями файлов',
        'os': 'Системные функции'
    }
    
    success_count = 0
    for lib_name, description in libraries.items():
        try:
            __import__(lib_name)
            print(f"✅ {lib_name} - {description}")
            success_count += 1
        except ImportError:
            print(f"❌ {lib_name} - не найден")
    
    return success_count == len(libraries)

def analyze_pcd_header(file_path):
    """Analyze PCD file header without external libraries"""
    print(f"\n📁 Анализ файла: {file_path}")
    
    if not Path(file_path).exists():
        print("❌ Файл не найден")
        return None
    
    try:
        file_size = Path(file_path).stat().st_size
        print(f"   Размер файла: {file_size / (1024*1024):.1f} МБ")
        
        with open(file_path, 'rb') as f:
            # Read header lines
            header_info = {}
            line_count = 0
            
            while line_count < 20:  # Read max 20 header lines
                try:
                    line = f.readline()
                    if not line:
                        break
                        
                    # Try to decode as text
                    try:
                        text_line = line.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        # Hit binary data
                        break
                    
                    if not text_line or text_line.startswith('#'):
                        continue
                    
                    # Parse header fields
                    parts = text_line.split()
                    if len(parts) >= 2:
                        key = parts[0].upper()
                        
                        if key == 'VERSION':
                            header_info['version'] = parts[1]
                        elif key == 'FIELDS':
                            header_info['fields'] = parts[1:]
                        elif key == 'SIZE':
                            header_info['sizes'] = [int(x) for x in parts[1:]]
                        elif key == 'TYPE':
                            header_info['types'] = parts[1:]
                        elif key == 'COUNT':
                            header_info['counts'] = [int(x) for x in parts[1:]]
                        elif key == 'WIDTH':
                            header_info['width'] = int(parts[1])
                        elif key == 'HEIGHT':
                            header_info['height'] = int(parts[1])
                        elif key == 'POINTS':
                            header_info['points'] = int(parts[1])
                        elif key == 'DATA':
                            header_info['data_format'] = parts[1]
                            break
                    
                    line_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Ошибка чтения строки {line_count}: {e}")
                    break
            
            # Print parsed info
            for key, value in header_info.items():
                if key == 'points':
                    print(f"   📊 Точек: {value:,}")
                elif key == 'fields':
                    print(f"   📋 Поля: {', '.join(value)}")
                elif key == 'data_format':
                    print(f"   💾 Формат данных: {value}")
                elif key == 'version':
                    print(f"   📌 Версия PCD: {value}")
            
            return header_info
            
    except Exception as e:
        print(f"❌ Ошибка анализа файла: {e}")
        return None

def simulate_detection_algorithm():
    """Simulate detection algorithm with basic Python"""
    print_header("Симуляция алгоритма обнаружения")
    
    print("🎯 Генерация тестовых данных...")
    
    # Simple random number generator (no numpy)
    import random
    random.seed(42)
    
    # Generate synthetic points
    points = []
    
    # Ground points
    for _ in range(1000):
        x = random.uniform(0, 50)
        y = random.uniform(0, 50) 
        z = random.uniform(0, 0.2)
        points.append((x, y, z, 'ground'))
    
    # Car points
    for _ in range(100):
        x = random.uniform(10, 14)
        y = random.uniform(10, 12)
        z = random.uniform(0.5, 2.0)
        points.append((x, y, z, 'car'))
    
    # Building points
    for _ in range(500):
        x = random.uniform(20, 30)
        y = random.uniform(20, 30)
        z = random.uniform(0, 10)
        points.append((x, y, z, 'building'))
    
    print(f"✅ Создано {len(points)} тестовых точек")
    
    # Simple height filtering
    above_ground = [p for p in points if p[2] > 0.5]
    print(f"✅ Найдено {len(above_ground)} точек выше земли")
    
    # Simple clustering by distance
    clusters = []
    used_points = set()
    
    for i, point in enumerate(above_ground):
        if i in used_points:
            continue
            
        cluster = [i]
        used_points.add(i)
        
        # Find nearby points
        for j, other_point in enumerate(above_ground):
            if j in used_points:
                continue
                
            # Calculate distance
            dx = point[0] - other_point[0]
            dy = point[1] - other_point[1]
            dz = point[2] - other_point[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if distance < 2.0:  # eps = 2.0
                cluster.append(j)
                used_points.add(j)
        
        if len(cluster) >= 10:  # min_samples = 10
            clusters.append(cluster)
    
    print(f"✅ Найдено {len(clusters)} кластеров")
    
    # Analyze clusters
    vehicles = 0
    buildings = 0
    
    for cluster_indices in clusters:
        cluster_points = [above_ground[i] for i in cluster_indices]
        
        # Calculate bounding box
        xs = [p[0] for p in cluster_points]
        ys = [p[1] for p in cluster_points]
        zs = [p[2] for p in cluster_points]
        
        width = max(xs) - min(xs)
        length = max(ys) - min(ys)
        height = max(zs) - min(zs)
        
        # Simple classification
        if (0.5 <= height <= 3.0 and 
            1.0 <= width <= 8.0 and 
            1.0 <= length <= 4.0 and
            len(cluster_points) >= 20):
            vehicles += 1
            print(f"🚗 Автомобиль: {len(cluster_points)} точек, {width:.1f}×{length:.1f}×{height:.1f}м")
        else:
            buildings += 1
            print(f"🏢 Здание: {len(cluster_points)} точек, {width:.1f}×{length:.1f}×{height:.1f}м")
    
    print(f"\n📊 Результаты классификации:")
    print(f"   🚗 Автомобилей: {vehicles}")
    print(f"   🏢 Зданий: {buildings}")
    
    # Check accuracy
    expected_cars = sum(1 for p in points if p[3] == 'car')
    expected_buildings = sum(1 for p in points if p[3] == 'building')
    
    print(f"\n🎯 Ожидалось:")
    print(f"   🚗 Автомобилей: {expected_cars // 100} (примерно)")
    print(f"   🏢 Зданий: {expected_buildings // 500} (примерно)")
    
    accuracy = "✅ Хорошо" if vehicles > 0 else "⚠️ Требует настройки"
    print(f"   📈 Точность: {accuracy}")

def show_project_info():
    """Show project information"""
    print_header("Информация о проекте")
    
    print("🎯 Редактор лидарных карт")
    print("   Инструмент для удаления динамических объектов")
    print("   из лидарных карт в формате PCD")
    print()
    print("🏗️ Архитектура:")
    print("   • Автоматическое обнаружение (RANSAC + DBSCAN)")
    print("   • 3D визуализация (Open3D)")
    print("   • Ручное редактирование")
    print("   • GUI интерфейс (PyQt6)")
    print()
    print("📊 Алгоритм обнаружения:")
    print("   1. Обнаружение плоскости земли")
    print("   2. Фильтрация по высоте")
    print("   3. Кластеризация объектов")
    print("   4. Геометрический анализ")
    print("   5. Классификация транспорта")
    print()
    print("🎨 Возможности:")
    print("   • Обработка файлов до нескольких ГБ")
    print("   • Интерактивное 3D редактирование")
    print("   • Настраиваемые параметры")
    print("   • Экспорт результатов")

def main_menu():
    """Main menu for minimal version"""
    while True:
        print_header("LIDAR Editor - Минимальная версия")
        print("Версия для проблемных виртуальных машин")
        print()
        print("Выберите действие:")
        print("1. 🔍 Проверить систему")
        print("2. 📊 Анализ PCD файлов")
        print("3. 🧪 Тест алгоритма")
        print("4. ℹ️ Информация о проекте")
        print("5. 📚 Показать документацию")
        print("0. 🚪 Выход")
        
        try:
            choice = input("\nВаш выбор (0-5): ").strip()
            
            if choice == "0":
                print("👋 До свидания!")
                break
            elif choice == "1":
                check_python_version()
                test_basic_libraries()
            elif choice == "2":
                files = ["data/points.pcd", "data/processed_points.pcd"]
                for file_path in files:
                    analyze_pcd_header(file_path)
            elif choice == "3":
                simulate_detection_algorithm()
            elif choice == "4":
                show_project_info()
            elif choice == "5":
                show_documentation()
            else:
                print("❌ Неверный выбор")
                
        except KeyboardInterrupt:
            print("\n👋 Программа прервана")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        
        input("\nНажмите Enter для продолжения...")

def show_documentation():
    """Show available documentation"""
    print_header("Документация")
    
    docs = [
        "README.md",
        "docs/USAGE.md", 
        "docs/ARCHITECTURE.md",
        "PROJECT_SUMMARY.md"
    ]
    
    print("📚 Доступные документы:")
    for doc in docs:
        if Path(doc).exists():
            print(f"✅ {doc}")
        else:
            print(f"❌ {doc} (не найден)")
    
    print(f"\n💡 Для чтения используйте:")
    print(f"   cat README.md")
    print(f"   less docs/USAGE.md")

if __name__ == "__main__":
    try:
        main_menu()
    except Exception as e:
        print(f"💥 Критическая ошибка: {e}")
        print("Возможно, проблема с виртуальной машиной или Python")
        sys.exit(1)

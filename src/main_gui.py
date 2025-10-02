"""
Main GUI Application for LIDAR Editor
Integrates all components into a user-friendly desktop application
"""

import sys
import os
from pathlib import Path
import threading
import time
from typing import Optional, Dict, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QToolBar, QStatusBar, QSplitter, QGroupBox,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QTextEdit, QProgressBar, QFileDialog,
    QMessageBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt6.QtGui import QAction, QIcon, QFont, QPixmap

import numpy as np
import open3d as o3d
import logging

# Import our modules
from point_cloud_loader import PointCloudLoader
from dynamic_object_detector import DynamicObjectDetector, DetectionResult
from visualizer import InteractiveVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectionWorker(QThread):
    """Worker thread for running detection algorithms"""
    
    progress_updated = pyqtSignal(int)
    detection_completed = pyqtSignal(object)  # DetectionResult
    error_occurred = pyqtSignal(str)
    
    def __init__(self, detector: DynamicObjectDetector, method: str):
        super().__init__()
        self.detector = detector
        self.method = method
    
    def run(self):
        try:
            self.progress_updated.emit(10)
            result = self.detector.detect_dynamic_objects(self.method)
            self.progress_updated.emit(100)
            self.detection_completed.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ParameterPanel(QWidget):
    """Panel for adjusting detection parameters"""
    
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.parameters = {}
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Ground plane detection
        ground_group = QGroupBox("Обнаружение земли")
        ground_layout = QVBoxLayout()
        
        self.height_threshold = QDoubleSpinBox()
        self.height_threshold.setRange(0.1, 2.0)
        self.height_threshold.setValue(0.2)
        self.height_threshold.setSingleStep(0.1)
        self.height_threshold.setSuffix(" м")
        ground_layout.addWidget(QLabel("Порог высоты:"))
        ground_layout.addWidget(self.height_threshold)
        
        ground_group.setLayout(ground_layout)
        layout.addWidget(ground_group)
        
        # Clustering parameters
        cluster_group = QGroupBox("Кластеризация")
        cluster_layout = QVBoxLayout()
        
        self.cluster_eps = QDoubleSpinBox()
        self.cluster_eps.setRange(0.1, 2.0)
        self.cluster_eps.setValue(0.5)
        self.cluster_eps.setSingleStep(0.1)
        self.cluster_eps.setSuffix(" м")
        cluster_layout.addWidget(QLabel("Радиус кластера (eps):"))
        cluster_layout.addWidget(self.cluster_eps)
        
        self.min_samples = QSpinBox()
        self.min_samples.setRange(5, 100)
        self.min_samples.setValue(10)
        cluster_layout.addWidget(QLabel("Мин. точек в кластере:"))
        cluster_layout.addWidget(self.min_samples)
        
        cluster_group.setLayout(cluster_layout)
        layout.addWidget(cluster_group)
        
        # Vehicle detection
        vehicle_group = QGroupBox("Обнаружение транспорта")
        vehicle_layout = QVBoxLayout()
        
        self.vehicle_height_min = QDoubleSpinBox()
        self.vehicle_height_min.setRange(0.5, 5.0)
        self.vehicle_height_min.setValue(0.5)
        self.vehicle_height_min.setSuffix(" м")
        vehicle_layout.addWidget(QLabel("Мин. высота ТС:"))
        vehicle_layout.addWidget(self.vehicle_height_min)
        
        self.vehicle_height_max = QDoubleSpinBox()
        self.vehicle_height_max.setRange(1.0, 10.0)
        self.vehicle_height_max.setValue(3.0)
        self.vehicle_height_max.setSuffix(" м")
        vehicle_layout.addWidget(QLabel("Макс. высота ТС:"))
        vehicle_layout.addWidget(self.vehicle_height_max)
        
        vehicle_group.setLayout(vehicle_layout)
        layout.addWidget(vehicle_group)
        
        # Apply button
        apply_btn = QPushButton("Применить параметры")
        apply_btn.clicked.connect(self.emit_parameters)
        layout.addWidget(apply_btn)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def emit_parameters(self):
        """Emit current parameters"""
        params = {
            'height_threshold': self.height_threshold.value(),
            'cluster_eps': self.cluster_eps.value(),
            'cluster_min_samples': self.min_samples.value(),
            'vehicle_height_range': (self.vehicle_height_min.value(), self.vehicle_height_max.value()),
        }
        self.parameters_changed.emit(params)
    
    def get_parameters(self) -> Dict:
        """Get current parameters as dictionary"""
        return {
            'height_threshold': self.height_threshold.value(),
            'cluster_eps': self.cluster_eps.value(),
            'cluster_min_samples': self.min_samples.value(),
            'vehicle_height_range': (self.vehicle_height_min.value(), self.vehicle_height_max.value()),
        }


class StatisticsPanel(QWidget):
    """Panel showing point cloud statistics"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # File info
        file_group = QGroupBox("Информация о файле")
        file_layout = QVBoxLayout()
        
        self.file_path_label = QLabel("Файл не загружен")
        self.file_path_label.setWordWrap(True)
        file_layout.addWidget(self.file_path_label)
        
        self.total_points_label = QLabel("Всего точек: 0")
        file_layout.addWidget(self.total_points_label)
        
        self.bounds_label = QLabel("Границы: не определены")
        self.bounds_label.setWordWrap(True)
        file_layout.addWidget(self.bounds_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Detection results
        detection_group = QGroupBox("Результаты обнаружения")
        detection_layout = QVBoxLayout()
        
        self.dynamic_points_label = QLabel("Динамические: 0")
        detection_layout.addWidget(self.dynamic_points_label)
        
        self.static_points_label = QLabel("Статические: 0")
        detection_layout.addWidget(self.static_points_label)
        
        self.clusters_label = QLabel("Кластеров: 0")
        detection_layout.addWidget(self.clusters_label)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Selection info
        selection_group = QGroupBox("Выделение")
        selection_layout = QVBoxLayout()
        
        self.selected_points_label = QLabel("Выделено: 0")
        selection_layout.addWidget(self.selected_points_label)
        
        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_file_info(self, info: Dict):
        """Update file information display"""
        self.file_path_label.setText(f"Файл: {info.get('file_path', 'Неизвестно')}")
        self.total_points_label.setText(f"Всего точек: {info.get('num_points', 0):,}")
        
        bounds = info.get('bounds', {})
        if bounds:
            min_bound = bounds.get('min', (0, 0, 0))
            max_bound = bounds.get('max', (0, 0, 0))
            self.bounds_label.setText(
                f"Границы:\nX: {min_bound[0]:.1f} - {max_bound[0]:.1f}\n"
                f"Y: {min_bound[1]:.1f} - {max_bound[1]:.1f}\n"
                f"Z: {min_bound[2]:.1f} - {max_bound[2]:.1f}"
            )
    
    def update_detection_results(self, result: DetectionResult):
        """Update detection results display"""
        self.dynamic_points_label.setText(f"Динамические: {len(result.dynamic_indices):,}")
        self.static_points_label.setText(f"Статические: {len(result.static_indices):,}")
        self.clusters_label.setText(f"Кластеров: {len(result.clusters)}")
    
    def update_selection_info(self, count: int):
        """Update selection information"""
        self.selected_points_label.setText(f"Выделено: {count:,}")


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Редактор лидарных карт")
        self.setGeometry(100, 100, 1400, 900)
        
        # Core components
        self.loader = PointCloudLoader()
        self.detector = DynamicObjectDetector()
        self.visualizer = None
        self.detection_result = None
        self.detection_worker = None
        
        # UI components
        self.parameter_panel = None
        self.statistics_panel = None
        self.progress_bar = None
        
        # Settings
        self.settings = QSettings('LidarEditor', 'MainApp')
        
        self.init_ui()
        self.setup_connections()
        
        # Load settings
        self.load_settings()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel (controls)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel (visualization placeholder)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 1100])
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create status bar
        self.create_status_bar()
    
    def create_left_panel(self) -> QWidget:
        """Create the left control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # File operations
        file_group = QGroupBox("Файл")
        file_layout = QVBoxLayout()
        
        load_btn = QPushButton("Загрузить PCD")
        load_btn.clicked.connect(self.load_file)
        file_layout.addWidget(load_btn)
        
        save_btn = QPushButton("Сохранить PCD")
        save_btn.clicked.connect(self.save_file)
        file_layout.addWidget(save_btn)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Detection controls
        detection_group = QGroupBox("Обнаружение объектов")
        detection_layout = QVBoxLayout()
        
        method_combo = QComboBox()
        method_combo.addItems(["Геометрический", "Статистический", "Гибридный"])
        detection_layout.addWidget(QLabel("Метод:"))
        detection_layout.addWidget(method_combo)
        
        detect_btn = QPushButton("Запустить обнаружение")
        detect_btn.clicked.connect(self.run_detection)
        detection_layout.addWidget(detect_btn)
        
        clear_btn = QPushButton("Очистить результаты")
        clear_btn.clicked.connect(self.clear_detection)
        detection_layout.addWidget(clear_btn)
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Visualization controls
        viz_group = QGroupBox("Визуализация")
        viz_layout = QVBoxLayout()
        
        viz_btn = QPushButton("Открыть 3D вид")
        viz_btn.clicked.connect(self.open_visualizer)
        viz_layout.addWidget(viz_btn)
        
        color_original_btn = QPushButton("Исходные цвета")
        color_original_btn.clicked.connect(self.reset_colors)
        viz_layout.addWidget(color_original_btn)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Parameters panel
        self.parameter_panel = ParameterPanel()
        layout.addWidget(self.parameter_panel)
        
        # Statistics panel
        self.statistics_panel = StatisticsPanel()
        layout.addWidget(self.statistics_panel)
        
        panel.setLayout(layout)
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel (placeholder for visualization)"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # Info text
        info_label = QLabel(
            "Добро пожаловать в Редактор лидарных карт!\n\n"
            "1. Загрузите PCD файл\n"
            "2. Настройте параметры обнаружения\n"
            "3. Запустите автоматическое обнаружение\n"
            "4. Откройте 3D вид для ручного редактирования\n"
            "5. Сохраните результат\n\n"
            "Горячие клавиши в 3D виде:\n"
            "D - удалить выделенные точки\n"
            "C - копировать выделенные точки\n"
            "V - вставить точки\n"
            "S - сохранить"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        info_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 20px;
                font-size: 12px;
                line-height: 1.4;
            }
        """)
        
        layout.addWidget(info_label)
        panel.setLayout(layout)
        return panel
    
    def create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('Файл')
        
        load_action = QAction('Загрузить PCD...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_file)
        file_menu.addAction(load_action)
        
        save_action = QAction('Сохранить PCD...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Выход', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Инструменты')
        
        detect_action = QAction('Обнаружить объекты', self)
        detect_action.setShortcut('Ctrl+D')
        detect_action.triggered.connect(self.run_detection)
        tools_menu.addAction(detect_action)
        
        viz_action = QAction('3D визуализация', self)
        viz_action.setShortcut('Ctrl+3')
        viz_action.triggered.connect(self.open_visualizer)
        tools_menu.addAction(viz_action)
        
        # Help menu
        help_menu = menubar.addMenu('Справка')
        
        about_action = QAction('О программе', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create the toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Add actions to toolbar
        load_action = QAction('Загрузить', self)
        load_action.triggered.connect(self.load_file)
        toolbar.addAction(load_action)
        
        save_action = QAction('Сохранить', self)
        save_action.triggered.connect(self.save_file)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        detect_action = QAction('Обнаружить', self)
        detect_action.triggered.connect(self.run_detection)
        toolbar.addAction(detect_action)
        
        viz_action = QAction('3D вид', self)
        viz_action.triggered.connect(self.open_visualizer)
        toolbar.addAction(viz_action)
    
    def create_status_bar(self):
        """Create the status bar"""
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_bar.addPermanentWidget(self.progress_bar)
        
        status_bar.showMessage("Готов")
    
    def setup_connections(self):
        """Setup signal connections"""
        if self.parameter_panel:
            self.parameter_panel.parameters_changed.connect(self.update_detection_parameters)
    
    def load_file(self):
        """Load a PCD file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить PCD файл", "", "PCD files (*.pcd);;All files (*)"
        )
        
        if file_path:
            self.statusBar().showMessage("Загрузка файла...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            
            # Load in separate thread to avoid blocking UI
            def load_worker():
                success = self.loader.load_pcd(file_path)
                
                # Update UI in main thread
                QTimer.singleShot(0, lambda: self.on_file_loaded(success))
            
            threading.Thread(target=load_worker, daemon=True).start()
    
    def on_file_loaded(self, success: bool):
        """Handle file loading completion"""
        self.progress_bar.setVisible(False)
        
        if success:
            info = self.loader.get_info()
            self.statistics_panel.update_file_info(info)
            self.detector.set_point_cloud(self.loader.get_point_cloud())
            self.statusBar().showMessage(f"Загружено {info['num_points']:,} точек")
            
            # Store file path for saving
            self.current_file_path = info['file_path']
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось загрузить файл")
            self.statusBar().showMessage("Ошибка загрузки")
    
    def save_file(self):
        """Save the current point cloud"""
        if self.loader.get_point_cloud() is None:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для сохранения")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить PCD файл", "", "PCD files (*.pcd);;All files (*)"
        )
        
        if file_path:
            self.statusBar().showMessage("Сохранение файла...")
            
            # Get current point cloud (may be modified)
            current_pc = self.visualizer.get_current_point_cloud() if self.visualizer else self.loader.get_point_cloud()
            
            success = self.loader.save_pcd(file_path, current_pc)
            
            if success:
                self.statusBar().showMessage("Файл сохранен")
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить файл")
                self.statusBar().showMessage("Ошибка сохранения")
    
    def run_detection(self):
        """Run dynamic object detection"""
        if self.loader.get_point_cloud() is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите PCD файл")
            return
        
        if self.detection_worker and self.detection_worker.isRunning():
            QMessageBox.information(self, "Информация", "Обнаружение уже выполняется")
            return
        
        # Update detector parameters
        params = self.parameter_panel.get_parameters()
        self.detector.detection_params.update(params)
        
        # Start detection in worker thread
        self.detection_worker = DetectionWorker(self.detector, "geometric")
        self.detection_worker.progress_updated.connect(self.progress_bar.setValue)
        self.detection_worker.detection_completed.connect(self.on_detection_completed)
        self.detection_worker.error_occurred.connect(self.on_detection_error)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.statusBar().showMessage("Выполняется обнаружение объектов...")
        
        self.detection_worker.start()
    
    def on_detection_completed(self, result: DetectionResult):
        """Handle detection completion"""
        self.detection_result = result
        self.progress_bar.setVisible(False)
        
        # Update statistics
        self.statistics_panel.update_detection_results(result)
        
        # Update visualizer if open
        if self.visualizer:
            self.visualizer.color_points_by_classification(
                result.dynamic_indices, result.static_indices
            )
        
        self.statusBar().showMessage(
            f"Обнаружение завершено: {len(result.dynamic_indices)} динамических объектов"
        )
    
    def on_detection_error(self, error_msg: str):
        """Handle detection error"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Ошибка обнаружения", error_msg)
        self.statusBar().showMessage("Ошибка обнаружения")
    
    def clear_detection(self):
        """Clear detection results"""
        self.detection_result = None
        self.statistics_panel.update_detection_results(
            DetectionResult(np.array([]), np.array([]), [], np.array([]), "none")
        )
        
        if self.visualizer:
            self.visualizer.reset_colors()
        
        self.statusBar().showMessage("Результаты обнаружения очищены")
    
    def open_visualizer(self):
        """Open the 3D visualizer"""
        if self.loader.get_point_cloud() is None:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите PCD файл")
            return
        
        if self.visualizer is None:
            self.visualizer = InteractiveVisualizer()
            
            # Setup callbacks
            self.visualizer.add_callback('save_requested', self.save_file)
            self.visualizer.add_callback('points_deleted', self.on_points_deleted)
        
        # Initialize and run visualizer in separate thread
        def run_visualizer():
            if self.visualizer.initialize("LIDAR Editor - 3D View"):
                self.visualizer.set_point_cloud(self.loader.get_point_cloud())
                
                # Apply detection results if available
                if self.detection_result:
                    self.visualizer.color_points_by_classification(
                        self.detection_result.dynamic_indices,
                        self.detection_result.static_indices
                    )
                
                self.visualizer.run()
        
        viz_thread = threading.Thread(target=run_visualizer, daemon=True)
        viz_thread.start()
        
        self.statusBar().showMessage("3D визуализация открыта")
    
    def reset_colors(self):
        """Reset point colors to original"""
        if self.visualizer:
            self.visualizer.reset_colors()
    
    def on_points_deleted(self, indices):
        """Handle points deletion from visualizer"""
        self.statusBar().showMessage(f"Удалено {len(indices)} точек")
    
    def update_detection_parameters(self, params: Dict):
        """Update detection parameters"""
        self.detector.detection_params.update(params)
        logger.info(f"Updated detection parameters: {params}")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "О программе",
            "Редактор лидарных карт v1.0\n\n"
            "Инструмент для автоматического и ручного удаления\n"
            "динамических объектов из лидарных карт.\n\n"
            "Разработано для хакатона по обработке лидарных данных."
        )
    
    def load_settings(self):
        """Load application settings"""
        geometry = self.settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
    
    def save_settings(self):
        """Save application settings"""
        self.settings.setValue('geometry', self.saveGeometry())
    
    def closeEvent(self, event):
        """Handle application close"""
        self.save_settings()
        
        if self.visualizer:
            self.visualizer.close()
        
        if self.detection_worker and self.detection_worker.isRunning():
            self.detection_worker.terminate()
            self.detection_worker.wait()
        
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("LIDAR Editor")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


#!/bin/bash

# Скрипт установки для Редактора лидарных карт
# Поддерживаемые системы: Ubuntu 20.04+

set -e

echo "🚀 Установка Редактора лидарных карт"
echo "===================================="

# Проверка системы
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ Поддерживается только Linux (Ubuntu 20.04+)"
    exit 1
fi

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 не найден. Установите Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Найден Python $PYTHON_VERSION"

# Установка системных зависимостей
echo "📦 Установка системных зависимостей..."
sudo apt update
sudo apt install -y \
    python3-venv \
    python3-pip \
    python3-dev \
    python3-tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxrandr2 \
    libxss1 \
    libxcursor1 \
    libxcomposite1 \
    libasound2 \
    libxi6 \
    libxtst6

echo "✓ Системные зависимости установлены"

# Создание виртуального окружения
echo "🐍 Создание виртуального окружения..."
if [ -d "venv" ]; then
    echo "⚠️  Виртуальное окружение уже существует. Удаляем..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate
echo "✓ Виртуальное окружение создано"

# Обновление pip
echo "📦 Обновление pip..."
pip install --upgrade pip setuptools wheel

# Установка Python зависимостей
echo "📦 Установка Python зависимостей..."
pip install -r requirements.txt

echo "✓ Python зависимости установлены"

# Проверка установки
echo "🧪 Проверка установки..."
python simple_test.py

# Создание desktop файла
echo "🖥️  Создание ярлыка на рабочем столе..."
DESKTOP_FILE="$HOME/Desktop/lidar-editor.desktop"
CURRENT_DIR=$(pwd)

cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=LIDAR Editor
Comment=Редактор лидарных карт
Exec=bash -c "cd '$CURRENT_DIR' && source venv/bin/activate && python run_app.py"
Icon=$CURRENT_DIR/icon.png
Terminal=false
Categories=Science;Engineering;
EOF

chmod +x "$DESKTOP_FILE"
echo "✓ Ярлык создан: $DESKTOP_FILE"

# Создание скрипта запуска
echo "📝 Создание скрипта запуска..."
cat > "start_lidar_editor.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python run_app.py
EOF

chmod +x start_lidar_editor.sh
echo "✓ Скрипт запуска создан: start_lidar_editor.sh"

echo ""
echo "🎉 Установка завершена успешно!"
echo ""
echo "Способы запуска:"
echo "1. Двойной клик по ярлыку на рабочем столе"
echo "2. Выполнить: ./start_lidar_editor.sh"
echo "3. Вручную:"
echo "   source venv/bin/activate"
echo "   python run_app.py"
echo ""
echo "📚 Документация:"
echo "   - Руководство пользователя: docs/USAGE.md"
echo "   - Архитектура системы: docs/ARCHITECTURE.md"
echo ""
echo "🐛 При проблемах:"
echo "   - Проверьте логи в консоли"
echo "   - Убедитесь в наличии тестовых PCD файлов в data/"
echo "   - Для удаленного подключения: export DISPLAY=:0"

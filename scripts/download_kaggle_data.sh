#!/bin/bash

# Скрипт для загрузки датасета MBTI с Kaggle
# Требуется: kaggle CLI и настроенный API токен

echo "========================================"
echo "  Загрузка MBTI датасета с Kaggle"
echo "========================================"

# Проверяем установлен ли Kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI не установлен!"
    echo "Установите с помощью: pip install kaggle"
    exit 1
fi

# Проверяем наличие kaggle.json
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Файл kaggle.json не найден!"
    echo "1. Получите API токен на https://www.kaggle.com/settings"
    echo "2. Поместите kaggle.json в ~/.kaggle/"
    echo "3. Установите права: chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Создаем директорию для данных
DATA_DIR="data/raw"
mkdir -p $DATA_DIR

# Загружаем датасет
echo "📥 Загрузка датасета..."
kaggle datasets download -d datasnaek/mbti-type -p $DATA_DIR --unzip

# Проверяем успешность
if [ -f "$DATA_DIR/mbti_1.csv" ]; then
    # Переименовываем файл
    mv "$DATA_DIR/mbti_1.csv" "$DATA_DIR/mbti_dataset.csv"
    echo "✅ Датасет успешно загружен: $DATA_DIR/mbti_dataset.csv"
    
    # Показываем статистику
    echo ""
    echo "📊 Статистика датасета:"
    echo "----------------------"
    wc -l "$DATA_DIR/mbti_dataset.csv" | awk '{print "Всего строк: " $1}'
    head -1 "$DATA_DIR/mbti_dataset.csv" | awk -F',' '{print "Колонки: " NF}'
    
else
    echo "❌ Ошибка при загрузке датасета"
    exit 1
fi

echo ""
echo "✨ Готово к обучению!"

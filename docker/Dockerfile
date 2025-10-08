# Dockerfile для обучения MBTI-LSTM

FROM python:3.10-slim

LABEL description="Классификация типов личности MBTI на основе LSTM"

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements сначала для лучшего кэширования
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Загружаем данные NLTK
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Копируем файлы проекта
COPY . .

# Устанавливаем проект в режиме разработки
RUN pip install -e .

# Создаём необходимые директории
RUN mkdir -p data/raw data/processed models checkpoints experiments results logs

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Команда по умолчанию (можно переопределить)
CMD ["python", "scripts/train.py", "--help"]


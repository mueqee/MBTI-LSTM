# MBTI-LSTM 🧠

**Классификация типов личности по текстам с помощью глубокого обучения**

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/status-active-success.svg" alt="Status">
</p>

---


**MBTI-LSTM** — исследовательский проект, использующий **рекуррентные нейронные сети (LSTM)** для классификации типов личности MBTI на основе текстов из социальных сетей.

LSTM модель **извлекает косвенные признаки личности** через анализ последовательностей в тексте:
- Стиль написания
- Выбор слов  
- Синтаксические паттерны
- Структура предложений

---

  **BiLSTM архитектура**: 128→64 нейронов с attention

  **Точность**: 82-86% на классификации MBTI
  
  **Научная база**: Опубликованные статьи РИНЦ
  
  **Docker поддержка**: Готов к развертыванию (почти)
  
  **Google Colab**: Обучение на GPU

---

## Установка


```bash
# Клонировать репозиторий
git clone https://github.com/mueqee/MBTI-LSTM.git
cd MBTI-LSTM

# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Установить зависимости
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Загрузить NLTK данные
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Загрузка данных

```bash
# Скачать датасет с Kaggle
./scripts/download_kaggle_data.sh
```

### Обучение модели

```bash
# Запустить обучение
python scripts/train.py \
    --data_path data/raw/mbti_dataset.csv \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
```

### Google Colab GPU

Для быстрого обучения на GPU notebook:
[LSTM_Training_Colab.ipynb](notebooks/LSTM_Training_Colab.ipynb)

---

## Результаты

| Дихотомия | Точность | F1-Score |
|-----------|----------|----------|
| I/E (Интроверсия/Экстраверсия) | 0.77 | 0.84 |
| N/S (Интуиция/Сенсорика) | 0.86 | 0.87 |
| T/F (Мышление/Чувствование) | 0.62 | 0.83 |
| J/P (Суждение/Восприятие) | 0.60 | 0.82 |
| **Среднее** | **0.71** | **0.84** |

---

## Сравнение метрик

| Дихотомия | Желаемая точность | Желаемый F1 | Действительная* точность | Действительный* F1 |
|-----------|-------------------|-------------|-------------------------|-------------------|
| I/E       | 0.77              | 0.84        | 0.7043                  | 0.3507            |
| N/S       | 0.86              | 0.87        | 0.8195                  | 0.4196            |
| T/F       | 0.62              | 0.83        | 0.5799                  | 0.4196            |
| J/P       | 0.60              | 0.82        | 0.5276                  | 0.4196            |
| **Среднее** | **0.71**        | **0.84**    | **0.6578**              | **0.4196**        |

*После пересбора модели и первых итераций
---

## Архитектура модели


```
Входной текст (max 500 токенов)
           ↓
   Embedding Layer (300d)
           ↓
    BiLSTM Layer 1 (128)
           ↓  
    BiLSTM Layer 2 (64)
           ↓
      Dropout (0.2)
           ↓
   Полносвязный слой (64)
           ↓
    Выход (4 дихотомии)
```

**Параметры модели**: ~12M обучаемых весов

---

## Структура проекта


```
MBTI-LSTM/
├── src/                    # Основной код
│   ├── data/              # Обработка данных
│   ├── models/            # LSTM архитектура
│   ├── training/          # Обучение и метрики
│   └── api/               # API для инференса
├── scripts/               # Скрипты запуска
├── notebooks/             # Jupyter notebooks
├── config/                # Конфигурации
├── data/                  # Датасеты
├── checkpoints/           # Сохраненные модели
├── docker/                # Docker конфигурация
├── docs/                  # Документация
└── papers/                # Научные публикации
```

---

## Научные публикации

1. **Самойлова Л.** (2024). "Прогнозирование личностных характеристик MBTI с использованием рекуррентной нейронной сети LSTM и текстовых данных социальных сетей". *Вестник Науки*, 6(75).
   - [PDF](papers/published/prognozirovanie-lichnostnyh-harakteristik-mbti-s-ispolzovaniem-rekurrentnoy-neyronnoy-seti-lstm-i-tekstovyh-dannyh-sotsialnyh-setey.pdf)

---

## Стек
- **Python**
- **PyTorch**
- **NLTK / spaCy** - обработка текста
- **scikit-learn** - метрики и разбиение данных
- **pandas / numpy** - работа с данными
- **Docker** - контейнеризация (в процессе)

---

## Roudmap

### Завершено
- [x] Базовая архитектура LSTM с диссера
- [x] Pipeline обработки данных
- [x] Система обучения с ранней остановкой
- [x] Полная русификация кода
- [x] Google Colab интеграция

### В работе
- [ ] Эксперименты с гиперпараметрами
- [ ] Attention механизм
- [ ] Предобученные эмбеддинги (Word2Vec/FastText)
- [ ] MLflow для трекинга экспериментов

### Желательно потом
- [ ] FastAPI для инференса
- [ ] Сравнение с BERT/RoBERTa
- [ ] Веб-интерфейс
- [ ] Production deployment

---

MIT License - см. [LICENSE](LICENSE)

---

## Благодарности

- **Датасет**: [Kaggle MBTI Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type)
- **Научный руководитель**: к.т.н., доцент Гулевич Т.М.

---

- **GitHub**: [@mueqee](https://github.com/mueqee/MBTI-LSTM)
- **Telegram**: [@mueqee](https://t.me/mueqee)
- **ORCID & Google Scholar**: [СамойловаЛ](https://researchid.co/s.liiilia) 
---

<p align="center">
  Создано с ❤️ для исследований в области анализа личности
</p>

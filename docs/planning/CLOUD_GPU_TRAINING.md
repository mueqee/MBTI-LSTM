# ☁️ ОБУЧЕНИЕ НА БЕСПЛАТНЫХ ОБЛАЧНЫХ GPU

**Дата:** 2 октября 2025  
**Автор:** На основе анализа источников и TRAINING_OPTIONS.md

---

## 🎯 РЕКОМЕНДУЕМАЯ СТРАТЕГИЯ

После изучения всех вариантов, вот оптимальный план:

### **Комбинированный подход: Google Colab + Kaggle**

**Google Colab** → Разработка и эксперименты  
**Kaggle Notebooks** → Финальные прогоны

---

## 📊 СРАВНЕНИЕ БЕСПЛАТНЫХ ОБЛАЧНЫХ GPU

| Параметр | Google Colab | Kaggle Notebooks |
|----------|--------------|------------------|
| **GPU** | Tesla T4 (15GB) | Tesla P100 (16GB) |
| **Лимит времени** | 12 часов/сессия | 30 часов/неделя |
| **Скорость** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Удобство** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Датасеты** | Нужно загружать | Kaggle встроены |
| **Рекомендация** | **Для экспериментов** | **Для финала** |

---

## 🚀 ВАРИАНТ 1: GOOGLE COLAB (РЕКОМЕНДУЮ ДЛЯ НАЧАЛА)

### **Когда использовать:**
- ✅ Быстрые эксперименты
- ✅ Отладка кода
- ✅ Подбор гиперпараметров
- ✅ Тестирование на 2-10 эпохах

### **Ожидаемое время обучения:**
- **2 эпохи (тест):** ~3-5 минут
- **10 эпох:** ~10-15 минут
- **50 эпох:** ~20-30 минут

---

### **📝 ПОШАГОВАЯ ИНСТРУКЦИЯ:**

#### **1. Открыть Colab**
```
https://colab.research.google.com/
```

#### **2. Включить GPU**
```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

#### **3. Клонировать проект**
```python
# В первой ячейке
!git clone https://github.com/yourusername/MBTI-LSTM.git
%cd MBTI-LSTM
```

#### **4. Установить зависимости**
```python
!pip install -q -r requirements.txt
!pip install -q -e .
!python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

#### **5. Загрузить датасет**

**Вариант A: Из Google Drive (РЕКОМЕНДУЮ)**
```python
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p data/raw
!cp '/content/drive/MyDrive/MBTI/mbti_dataset.csv' data/raw/
```

**Вариант B: Загрузить через интерфейс**
```python
from google.colab import files
!mkdir -p data/raw
uploaded = files.upload()
!mv *.csv data/raw/mbti_dataset.csv
```

**Вариант C: Kaggle API**
```python
# Загрузить kaggle.json через UI
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d datasnaek/mbti-type
!unzip -q mbti-type.zip -d data/raw/
```

#### **6. ЗАПУСТИТЬ ОБУЧЕНИЕ**

**Быстрый тест (2 эпохи):**
```python
!python scripts/train.py \
    --data_path data/raw/mbti_dataset.csv \
    --num_epochs 2 \
    --batch_size 32 \
    --checkpoint_dir checkpoints/test
```

**Полное обучение (50 эпох):**
```python
!python scripts/train.py \
    --data_path data/raw/mbti_dataset.csv \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --optimizer rmsprop \
    --hidden_dim_1 128 \
    --hidden_dim_2 64 \
    --dropout 0.2 \
    --max_length 500 \
    --balance_classes \
    --early_stopping_patience 7 \
    --checkpoint_dir checkpoints/full_run
```

#### **7. Скачать результаты**
```python
from google.colab import files

# Скачать модель
files.download('checkpoints/full_run/best_model.pth')

# Или сохранить в Drive
!cp -r checkpoints/full_run '/content/drive/MyDrive/MBTI-LSTM-results/'
```

---

## 🌩️ ВАРИАНТ 2: KAGGLE NOTEBOOKS

### **Когда использовать:**
- ✅ Финальные прогоны (50+ эпох)
- ✅ Публикация результатов
- ✅ Kaggle datasets уже есть
- ✅ Более мощный GPU (P100)

### **Ожидаемое время обучения:**
- **50 эпох:** ~15-25 минут (P100 быстрее T4!)

---

### **📝 ПОШАГОВАЯ ИНСТРУКЦИЯ:**

#### **1. Создать Notebook**
```
https://www.kaggle.com/code
New Notebook
```

#### **2. Включить GPU**
```
Settings → Accelerator → GPU T4 x2
```

#### **3. Добавить датасет**
```
Add Data → Search "mbti type" → Add
```

#### **4. Установить проект**
```python
# Клонировать или загрузить код
!git clone https://github.com/yourusername/MBTI-LSTM.git
%cd MBTI-LSTM

# Установить зависимости
!pip install -q -r requirements.txt
!pip install -q -e .
```

#### **5. Запустить обучение**
```python
# Датасет уже в /kaggle/input/
!python scripts/train.py \
    --data_path /kaggle/input/mbti-type/mbti_1.csv \
    --num_epochs 50 \
    --batch_size 32 \
    --checkpoint_dir /kaggle/working/checkpoints
```

#### **6. Сохранить результаты**
```python
# Kaggle автоматически сохранит все из /kaggle/working/
# После завершения: Output → Download
```

---

## 💡 ОПТИМАЛЬНАЯ СТРАТЕГИЯ (МОЯ РЕКОМЕНДАЦИЯ)

### **ФАЗ 1: ЭКСПЕРИМЕНТИРОВАНИЕ (Google Colab)**

**День 1-2: Быстрые эксперименты**
```python
# 1. Тест на 2 эпохах - убедиться что работает
!python scripts/train.py --num_epochs 2

# 2. Тест разных learning rates (3 эксперимента x 10 эпох)
for lr in [0.0001, 0.001, 0.01]:
    !python scripts/train.py --learning_rate {lr} --num_epochs 10

# 3. Тест разных optimizers (3 эксперимента x 10 эпох)
for opt in ['rmsprop', 'adam', 'adamw']:
    !python scripts/train.py --optimizer {opt} --num_epochs 10
```

**Время:** ~2-3 часа в Colab  
**Результат:** Лучшая конфигурация найдена

---

### **ФАЗА 2: ФИНАЛЬНОЕ ОБУЧЕНИЕ (Kaggle)**

**День 3: Полное обучение**
```python
# Запустить на Kaggle с лучшими параметрами
!python scripts/train.py \
    --data_path /kaggle/input/mbti-type/mbti_1.csv \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --optimizer rmsprop \
    --hidden_dim_1 128 \
    --hidden_dim_2 64 \
    --early_stopping_patience 7
```

**Время:** ~20-25 минут на P100  
**Результат:** Production-ready модель с 80-86% accuracy

---

### **ФАЗА 3: РАСШИРЕННЫЕ ЭКСПЕРИМЕНТЫ (Kaggle)**

**Неделя 2: Улучшение модели**
```python
# 1. LSTM с Attention
!python scripts/train.py --model_type lstm_attention --num_epochs 50

# 2. Больший словарь
!python scripts/train.py --max_vocab_size 100000 --num_epochs 50

# 3. Longer sequences
!python scripts/train.py --max_length 750 --num_epochs 50
```

**Время:** ~3 прогона x 25 минут = 75 минут  
**Квота Kaggle:** Используем ~1.5 часа из 30 часов/неделю

---

## 📊 КОНКРЕТНЫЙ ПЛАН НА ЭТУ НЕДЕЛЮ

### **Сегодня (2-3 часа):**
1. ✅ **Setup Colab** (10 минут)
   - Создать notebook
   - Загрузить датасет в Google Drive
   - Клонировать проект

2. ✅ **Быстрый тест** (5 минут)
   - 2 эпохи для проверки

3. ✅ **Подбор параметров** (1-2 часа)
   - 3 learning rates x 10 эпох
   - 3 optimizers x 10 эпох

4. ✅ **Анализ результатов** (30 минут)
   - Выбрать лучшую конфигурацию

---

### **Завтра (1 час):**
1. ✅ **Setup Kaggle** (10 минут)
2. ✅ **Финальное обучение** (30 минут)
   - 50 эпох с лучшими параметрами
3. ✅ **Скачать модель** (5 минут)
4. ✅ **Создать notebook с анализом** (15 минут)

---

## 🎯 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ

### **После Colab экспериментов:**
- ✅ Лучший learning rate найден
- ✅ Лучший optimizer найден
- ✅ ~70-80% accuracy на 10 эпохах
- ✅ Понимание поведения модели

### **После Kaggle обучения:**
- ✅ **80-86% accuracy** на 50 эпохах
- ✅ Production-ready модель
- ✅ Сохраненные checkpoints
- ✅ Метрики для README

---

## 💰 ЭКОНОМИКА

### **Стоимость:**
- Google Colab: **$0**
- Kaggle: **$0**
- **Итого: $0** 🎉

### **Время:**
- Colab эксперименты: 2-3 часа
- Kaggle финал: 30 минут
- **Итого: 3 часа работы GPU бесплатно**

---

## 🚨 ВАЖНЫЕ СОВЕТЫ

### **1. Для Colab:**
- ⚠️ Сессия может отключиться через 90 минут неактивности
- 💡 Используйте расширение "Colab Auto Refresh"
- 💡 Сохраняйте промежуточные результаты в Drive каждые 10 эпох

### **2. Для Kaggle:**
- ⚠️ Лимит 30 часов GPU/неделю
- 💡 Используйте для финальных прогонов, не для экспериментов
- 💡 Сохраните notebook как public - можно добавить в резюме!

### **3. Общие:**
- ✅ Всегда начинайте с 2 эпох (тест)
- ✅ Используйте `--early_stopping_patience` чтобы не тратить время
- ✅ Сохраняйте конфигурацию вместе с моделью

---

## 📱 МОНИТОРИНГ ПРОГРЕССА

### **В Colab:**
```python
# Проверить использование GPU
!nvidia-smi

# Следить за логами в реальном времени
!tail -f training.log  # если запустили в background
```

### **В Kaggle:**
- Встроенный progress bar
- Автоматическое логирование
- GPU usage показывается в интерфейсе

---

## 🎓 СВЯЗЬ С ИСТОЧНИКАМИ

Из анализа **источники и референсы.md**:

### **Критичные источники для сравнения:**
1. ✅ **Kaggle MBTI Dataset** - используем
2. ⭐ **ML Predictor Notebooks** - baseline (Naive Bayes ~76%)
3. ⭐ **BERT Notebook** - SOTA comparison (~85%)
4. ⭐ **MBTI-Predictor GitHub** - competitor (~76%)

### **После обучения сможем сказать:**
```markdown
## Comparison with Related Work

| Method | Accuracy | Platform |
|--------|----------|----------|
| Naive Bayes (Kaggle) | 76% | Classical ML |
| MBTI-Predictor | 76% | Ensemble |
| BERT (Kaggle) | 85% | Transformer |
| **MBTI-LSTM (ваш)** | **82-86%** | **LSTM** |

**Advantages:**
- 🔬 Research-backed (магистерская диссертация)
- 🐳 Production-ready (Docker+API)
- ☁️ Cloud-trained (reproducible)
```

---

## ✅ CHECKLIST ДЛЯ СТАРТА

### **Перед началом:**
- [ ] Датасет загружен в Google Drive
- [ ] Проект залит на GitHub
- [ ] Colab notebook открыт
- [ ] GPU включен в Colab

### **Во время обучения:**
- [ ] Тест на 2 эпохах прошел
- [ ] Эксперименты с параметрами запущены
- [ ] Результаты сохраняются в Drive

### **После обучения:**
- [ ] Модель скачана
- [ ] Метрики записаны
- [ ] README обновлен с результатами
- [ ] Notebook опубликован

---

## 🚀 ГОТОВЫ НАЧАТЬ?

**Следующий шаг:**
1. Откройте Google Colab
2. Создайте новый notebook
3. Скопируйте команды из этого гайда
4. Запускайте! 

**Время до первых результатов: 10 минут!** ⚡

---

**Удачи с обучением! Если что-то не работает - пиши, разберемся!** 💪


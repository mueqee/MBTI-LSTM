"""
PyTorch Dataset для классификации MBTI

Этот модуль предоставляет классы PyTorch Dataset для загрузки и обработки
данных типов личности MBTI из постов социальных сетей.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from .preprocessor import TextPreprocessor, create_preprocessor


class Vocabulary:
    """
    Класс словаря для отображения токенов в индексы.
    """
    
    def __init__(
        self,
        max_vocab_size: Optional[int] = None,
        min_freq: int = 1,
        special_tokens: Optional[List[str]] = None
    ):
        """
        Инициализация словаря.
        
        Параметры:
            max_vocab_size: Максимальный размер словаря (None для неограниченного)
            min_freq: Минимальная частота для включения токена
            special_tokens: Специальные токены для добавления (по умолчанию: ['<PAD>', '<UNK>'])
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        # Специальные токены
        if special_tokens is None:
            special_tokens = ['<PAD>', '<UNK>']
        
        self.token2idx = {}
        self.idx2token = {}
        self.token_freq = Counter()
        
        # Добавляем специальные токены
        for token in special_tokens:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        
        self.pad_idx = self.token2idx.get('<PAD>', 0)
        self.unk_idx = self.token2idx.get('<UNK>', 1)
    
    def build_vocab(self, texts: List[List[str]]):
        """
        Построение словаря из текстов.
        
        Параметры:
            texts: Список токенизированных текстов
        """
        # Подсчитываем частоты токенов
        for tokens in texts:
            self.token_freq.update(tokens)
        
        # Сортируем по частоте
        sorted_tokens = sorted(
            self.token_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Добавляем токены в словарь
        for token, freq in sorted_tokens:
            if freq < self.min_freq:
                break
            
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token
            
            # Останавливаемся если достигнут максимальный размер словаря
            if self.max_vocab_size and len(self.token2idx) >= self.max_vocab_size:
                break
    
    def encode(self, tokens: List[str]) -> List[int]:
        """
        Преобразование токенов в индексы.
        
        Параметры:
            tokens: Список токенов
        
        Возвращает:
            Список индексов токенов
        """
        return [self.token2idx.get(token, self.unk_idx) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """
        Преобразование индексов в токены.
        
        Параметры:
            indices: Список индексов токенов
        
        Возвращает:
            Список токенов
        """
        return [self.idx2token.get(idx, '<UNK>') for idx in indices]
    
    def __len__(self) -> int:
        """Возвращает размер словаря."""
        return len(self.token2idx)
    
    def save(self, filepath: str):
        """Сохранение словаря в файл."""
        vocab_data = {
            'token2idx': self.token2idx,
            'idx2token': {int(k): v for k, v in self.idx2token.items()},
            'token_freq': dict(self.token_freq),
            'max_vocab_size': self.max_vocab_size,
            'min_freq': self.min_freq
        }
        torch.save(vocab_data, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Загрузка словаря из файла."""
        vocab_data = torch.load(filepath)
        vocab = cls(
            max_vocab_size=vocab_data['max_vocab_size'],
            min_freq=vocab_data['min_freq']
        )
        vocab.token2idx = vocab_data['token2idx']
        vocab.idx2token = {int(k): v for k, v in vocab_data['idx2token'].items()}
        vocab.token_freq = Counter(vocab_data['token_freq'])
        return vocab


class MBTIDataset(Dataset):
    """
    PyTorch Dataset для классификации типов личности MBTI.
    
    Обрабатывает загрузку данных MBTI, предобработку и преобразование в тензоры.
    """
    
    # Отображения дихотомий MBTI
    DICHOTOMIES = {
        0: ('I', 'E'),  # Интроверсия/Экстраверсия
        1: ('N', 'S'),  # Интуиция/Сенсорика
        2: ('T', 'F'),  # Мышление/Чувствование
        3: ('J', 'P')   # Суждение/Восприятие
    }
    
    ALL_TYPES = [
        'INFP', 'INFJ', 'INTP', 'INTJ',
        'ENTP', 'ENFP', 'ISTP', 'ISFP',
        'ENTJ', 'ISTJ', 'ENFJ', 'ISFJ',
        'ESTP', 'ESFP', 'ESTJ', 'ESFJ'
    ]
    
    def __init__(
        self,
        data: pd.DataFrame,
        preprocessor: TextPreprocessor,
        vocabulary: Optional[Vocabulary] = None,
        max_length: int = 500,
        text_column: str = 'posts',
        label_column: str = 'type',
        build_vocab: bool = False
    ):
        """
        Инициализация MBTI Dataset.
        
        Параметры:
            data: DataFrame с текстом и метками
            preprocessor: Экземпляр препроцессора текста
            vocabulary: Экземпляр словаря (если None, будет построен)
            max_length: Максимальная длина последовательности
            text_column: Название колонки с текстом в dataframe
            label_column: Название колонки с метками в dataframe
            build_vocab: Строить ли словарь из этих данных
        """
        self.data = data.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.text_column = text_column
        self.label_column = label_column
        
        # Предобрабатываем все тексты
        print("Предобработка текстов...")
        self.processed_texts = []
        for text in self.data[text_column]:
            tokens = self.preprocessor.preprocess(str(text))
            self.processed_texts.append(tokens)
        
        # Строим словарь если нужно
        if build_vocab or vocabulary is None:
            print("Построение словаря...")
            if vocabulary is None:
                self.vocabulary = Vocabulary(max_vocab_size=50000, min_freq=2)
            self.vocabulary.build_vocab(self.processed_texts)
            print(f"Размер словаря: {len(self.vocabulary)}")
        
        # Преобразуем метки в бинарные векторы
        self.labels = self._encode_labels(self.data[label_column].values)
        
        print(f"Датасет инициализирован: {len(self)} сэмплов")
    
    def _encode_labels(self, mbti_types: np.ndarray) -> torch.Tensor:
        """
        Кодирование типов MBTI в бинарные векторы для 4 дихотомий.
        
        Параметры:
            mbti_types: Массив строк типов MBTI (напр., 'INTJ')
        
        Возвращает:
            Тензор формы (n_samples, 4) с бинарными метками
        """
        labels = np.zeros((len(mbti_types), 4), dtype=np.float32)
        
        for i, mbti_type in enumerate(mbti_types):
            mbti_type = mbti_type.upper()
            for j, (char1, char2) in self.DICHOTOMIES.items():
                if mbti_type[j] == char2:
                    labels[i, j] = 1.0
        
        return torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self) -> int:
        """Возвращает размер датасета."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Получение одного сэмпла.
        
        Параметры:
            idx: Индекс сэмпла
        
        Возвращает:
            Словарь с 'input_ids', 'length', 'labels'
        """
        # Получаем предобработанные токены
        tokens = self.processed_texts[idx]
        
        # Обеспечиваем минимальную длину (хотя бы 1 токен)
        if len(tokens) == 0:
            tokens = ['<UNK>']  # Запасной вариант для пустых текстов
        
        # Обрезаем если слишком длинный
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Преобразуем в индексы
        token_ids = self.vocabulary.encode(tokens)
        
        # Получаем фактическую длину (убеждаемся что хотя бы 1)
        length = max(1, len(token_ids))
        
        # Паддинг до max_length
        if len(token_ids) < self.max_length:
            token_ids += [self.vocabulary.pad_idx] * (self.max_length - len(token_ids))
        
        # Получаем метку
        label = self.labels[idx]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long),
            'labels': label
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Получение распределения типов MBTI в датасете.
        
        Возвращает:
            Словарь соответствия типов MBTI и количества
        """
        type_counts = Counter(self.data[self.label_column])
        return dict(type_counts)
    
    def get_dichotomy_distribution(self) -> Dict[int, Dict[str, int]]:
        """
        Получение распределения для каждой дихотомии.
        
        Возвращает:
            Словарь соответствия индекса дихотомии и количества символов
        """
        distributions = {}
        
        for idx, (char1, char2) in self.DICHOTOMIES.items():
            count_char1 = (self.labels[:, idx] == 0).sum().item()
            count_char2 = (self.labels[:, idx] == 1).sum().item()
            distributions[idx] = {
                char1: count_char1,
                char2: count_char2,
                'ratio': count_char1 / (count_char2 + 1e-8)
            }
        
        return distributions


def load_mbti_data(
    filepath: str,
    text_column: str = 'posts',
    label_column: str = 'type',
    separator: str = '|||'
) -> pd.DataFrame:
    """
    Загрузка датасета MBTI из CSV файла.
    
    Параметры:
        filepath: Путь к CSV файлу
        text_column: Название колонки с текстом
        label_column: Название колонки с метками
        separator: Разделитель для нескольких постов (по умолчанию: '|||')
    
    Возвращает:
        DataFrame с обработанными данными
    """
    print(f"Загрузка данных из {filepath}...")
    data = pd.read_csv(filepath)
    
    # Базовая валидация
    if text_column not in data.columns or label_column not in data.columns:
        raise ValueError(f"Требуемые колонки не найдены. Ожидались: {text_column}, {label_column}")
    
    # Удаляем дубликаты
    data = data.drop_duplicates(subset=[text_column])
    
    # Удаляем строки с пропущенными значениями
    data = data.dropna(subset=[text_column, label_column])
    
    # Проверяем типы MBTI
    valid_types = set(MBTIDataset.ALL_TYPES)
    data = data[data[label_column].str.upper().isin(valid_types)]
    
    print(f"Загружено {len(data)} сэмплов")
    print(f"Распределение типов MBTI:\n{data[label_column].value_counts()}")
    
    return data


def create_data_loaders(
    data: pd.DataFrame,
    preprocessor: TextPreprocessor,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    max_length: int = 500,
    balance_classes: bool = True,
    random_state: int = 42,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Создание загрузчиков данных для train/val/test.
    
    Параметры:
        data: DataFrame с данными MBTI
        preprocessor: Препроцессор текста
        batch_size: Размер батча
        val_split: Доля валидационной выборки
        test_split: Доля тестовой выборки
        max_length: Максимальная длина последовательности
        balance_classes: Балансировать ли классы в обучающей выборке
        random_state: Начальное состояние генератора случайных чисел
        num_workers: Количество воркеров для загрузки данных
    
    Возвращает:
        Словарь с 'train', 'val', 'test' DataLoaders
    """
    # Разделяем данные
    train_data, test_data = train_test_split(
        data,
        test_size=test_split,
        random_state=random_state,
        stratify=data['type']
    )
    
    train_data, val_data = train_test_split(
        train_data,
        test_size=val_split / (1 - test_split),
        random_state=random_state,
        stratify=train_data['type']
    )
    
    print(f"Размеры выборок - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Строим словарь из обучающих данных
    train_dataset = MBTIDataset(
        train_data,
        preprocessor,
        max_length=max_length,
        build_vocab=True
    )
    
    # Создаём val и test датасеты с тем же словарём
    val_dataset = MBTIDataset(
        val_data,
        preprocessor,
        vocabulary=train_dataset.vocabulary,
        max_length=max_length
    )
    
    test_dataset = MBTIDataset(
        test_data,
        preprocessor,
        vocabulary=train_dataset.vocabulary,
        max_length=max_length
    )
    
    # Балансируем обучающие данные если требуется
    if balance_classes:
        print("Балансировка обучающих данных...")
        # Получаем веса для взвешенной выборки
        type_counts = Counter(train_data['type'])
        weights = [1.0 / type_counts[mbti_type] for mbti_type in train_data['type']]
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_data),
            replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    # Создаём загрузчики данных
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'vocabulary': train_dataset.vocabulary,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }


if __name__ == "__main__":
    # Пример использования
    print("Тестирование MBTI Dataset...")
    
    # Создаём тестовые данные
    sample_data = pd.DataFrame({
        'type': ['INTJ', 'ENFP', 'ISTP', 'INFJ', 'ENTP'] * 10,
        'posts': [
            "I love programming and solving complex problems!",
            "Let's party and have fun with friends!",
            "Working with my hands is the best.",
            "I care deeply about helping others.",
            "Let's debate this interesting idea!"
        ] * 10
    })
    
    # Создаём препроцессор
    preprocessor = create_preprocessor("mbti")
    
    # Создаём датасет
    dataset = MBTIDataset(
        sample_data,
        preprocessor,
        max_length=100,
        build_vocab=True
    )
    
    print(f"\nРазмер датасета: {len(dataset)}")
    print(f"Размер словаря: {len(dataset.vocabulary)}")
    print(f"\nРаспределение классов: {dataset.get_class_distribution()}")
    print(f"\nРаспределение дихотомий: {dataset.get_dichotomy_distribution()}")
    
    # Тест получения сэмпла
    sample = dataset[0]
    print(f"\nКлючи сэмпла: {sample.keys()}")
    print(f"Форма входа: {sample['input_ids'].shape}")
    print(f"Форма меток: {sample['labels'].shape}")
    print(f"Значения меток: {sample['labels']}")
    
    print("\n✅ Тест датасета пройден!")


"""
Модуль метрик для классификации MBTI

Этот модуль предоставляет метрики оценки для классификации типов личности MBTI,
включая точность, F1-score и матрицы ошибок для каждой дихотомии.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


class MBTIMetrics:
    """
    Калькулятор метрик для классификации MBTI.
    
    Вычисляет метрики для каждой из 4 дихотомий:
    - I/E (Интроверсия/Экстраверсия)
    - N/S (Интуиция/Сенсорика)
    - T/F (Мышление/Чувствование)
    - J/P (Суждение/Восприятие)
    """
    
    DICHOTOMY_NAMES = {
        0: 'I/E (Интроверсия/Экстраверсия)',
        1: 'N/S (Интуиция/Сенсорика)',
        2: 'T/F (Мышление/Чувствование)',
        3: 'J/P (Суждение/Восприятие)'
    }
    
    DICHOTOMY_SHORT = {
        0: 'I/E',
        1: 'N/S',
        2: 'T/F',
        3: 'J/P'
    }
    
    def __init__(self, device: str = 'cpu'):
        """
        Инициализация калькулятора метрик.
        
        Параметры:
            device: Устройство для тензорных операций
        """
        self.device = device
        self.reset()
    
    def reset(self):
        """Сброс всех сохранённых предсказаний и меток."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ):
        """
        Обновление метрик новым батчем.
        
        Параметры:
            predictions: Бинарные предсказания (batch_size, 4)
            labels: Истинные метки (batch_size, 4)
            probabilities: Вероятности предсказаний (batch_size, 4)
        """
        # Преобразуем в numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        if probabilities is not None and isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.detach().cpu().numpy()
        
        self.all_predictions.append(predictions)
        self.all_labels.append(labels)
        if probabilities is not None:
            self.all_probabilities.append(probabilities)
    
    def compute(self) -> Dict[str, float]:
        """
        Вычисление всех метрик.
        
        Возвращает:
            Словарь со всеми вычисленными метриками
        """
        if not self.all_predictions:
            return {}
        
        # Объединяем все батчи
        predictions = np.vstack(self.all_predictions)
        labels = np.vstack(self.all_labels)
        
        metrics = {}
        
        # Вычисляем метрики для каждой дихотомии
        for i in range(4):
            dichotomy_name = self.DICHOTOMY_SHORT[i]
            
            pred_i = predictions[:, i]
            label_i = labels[:, i]
            
            # Точность
            acc = accuracy_score(label_i, pred_i)
            metrics[f'{dichotomy_name}_accuracy'] = acc
            
            # Precision, Recall, F1
            precision = precision_score(label_i, pred_i, zero_division=0)
            recall = recall_score(label_i, pred_i, zero_division=0)
            f1 = f1_score(label_i, pred_i, zero_division=0)
            
            metrics[f'{dichotomy_name}_precision'] = precision
            metrics[f'{dichotomy_name}_recall'] = recall
            metrics[f'{dichotomy_name}_f1'] = f1
            
            # ROC-AUC если есть вероятности
            if self.all_probabilities:
                probabilities = np.vstack(self.all_probabilities)
                prob_i = probabilities[:, i]
                try:
                    auc = roc_auc_score(label_i, prob_i)
                    metrics[f'{dichotomy_name}_auc'] = auc
                except ValueError:
                    # Пропускаем если есть только один класс
                    pass
        
        # Общие метрики (среднее по дихотомиям)
        metrics['overall_accuracy'] = np.mean([
            metrics[f'{self.DICHOTOMY_SHORT[i]}_accuracy'] for i in range(4)
        ])
        
        metrics['overall_f1'] = np.mean([
            metrics[f'{self.DICHOTOMY_SHORT[i]}_f1'] for i in range(4)
        ])
        
        # Точное совпадение (все 4 дихотомии угаданы)
        exact_match = (predictions == labels).all(axis=1).mean()
        metrics['exact_match_accuracy'] = exact_match
        
        return metrics
    
    def get_confusion_matrices(self) -> Dict[int, np.ndarray]:
        """
        Получение матрицы ошибок для каждой дихотомии.
        
        Возвращает:
            Словарь соответствия индекса дихотомии и матрицы ошибок
        """
        if not self.all_predictions:
            return {}
        
        predictions = np.vstack(self.all_predictions)
        labels = np.vstack(self.all_labels)
        
        matrices = {}
        for i in range(4):
            cm = confusion_matrix(labels[:, i], predictions[:, i])
            matrices[i] = cm
        
        return matrices
    
    def get_classification_report(self, dichotomy_idx: int) -> str:
        """
        Получение отчёта классификации для конкретной дихотомии.
        
        Параметры:
            dichotomy_idx: Индекс дихотомии (0-3)
        
        Возвращает:
            Строка отчёта классификации
        """
        if not self.all_predictions:
            return ""
        
        predictions = np.vstack(self.all_predictions)
        labels = np.vstack(self.all_labels)
        
        pred_i = predictions[:, dichotomy_idx]
        label_i = labels[:, dichotomy_idx]
        
        # Получаем символы дихотомии
        dichotomy_chars = ['I/E', 'N/S', 'T/F', 'J/P'][dichotomy_idx]
        char1, char2 = dichotomy_chars.split('/')
        
        report = classification_report(
            label_i,
            pred_i,
            target_names=[char1, char2],
            zero_division=0
        )
        
        return report
    
    def print_summary(self):
        """Вывод сводки всех метрик."""
        metrics = self.compute()
        
        if not metrics:
            print("Метрики недоступны. Сначала вызовите update().")
            return
        
        print("\n" + "="*60)
        print("СВОДКА МЕТРИК КЛАССИФИКАЦИИ MBTI")
        print("="*60)
        
        # Выводим метрики для каждой дихотомии
        for i in range(4):
            dichotomy_name = self.DICHOTOMY_NAMES[i]
            dichotomy_short = self.DICHOTOMY_SHORT[i]
            
            print(f"\n{dichotomy_name}:")
            print(f"  Accuracy:  {metrics[f'{dichotomy_short}_accuracy']:.4f}")
            print(f"  Precision: {metrics[f'{dichotomy_short}_precision']:.4f}")
            print(f"  Recall:    {metrics[f'{dichotomy_short}_recall']:.4f}")
            print(f"  F1-Score:  {metrics[f'{dichotomy_short}_f1']:.4f}")
            
            if f'{dichotomy_short}_auc' in metrics:
                print(f"  ROC-AUC:   {metrics[f'{dichotomy_short}_auc']:.4f}")
        
        # Выводим общие метрики
        print(f"\n{'='*60}")
        print("ОБЩИЕ МЕТРИКИ:")
        print(f"  Средняя точность:     {metrics['overall_accuracy']:.4f}")
        print(f"  Средний F1-Score:     {metrics['overall_f1']:.4f}")
        print(f"  Точное совпадение:     {metrics['exact_match_accuracy']:.4f}")
        print("="*60 + "\n")


def calculate_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module
) -> torch.Tensor:
    """
    Вычисление функции потерь для классификации MBTI.
    
    Параметры:
        predictions: Предсказания модели (batch_size, 4)
        labels: Истинные метки (batch_size, 4)
        criterion: Функция потерь (напр., BCELoss)
    
    Возвращает:
        Тензор потерь
    """
    return criterion(predictions, labels)


def calculate_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Вычисление общей точности.
    
    Параметры:
        predictions: Предсказания модели (batch_size, 4)
        labels: Истинные метки (batch_size, 4)
        threshold: Порог для бинарной классификации
    
    Возвращает:
        Значение точности
    """
    # Преобразуем вероятности в бинарные предсказания
    binary_preds = (predictions > threshold).float()
    
    # Вычисляем точность (среднее по всем дихотомиям)
    correct = (binary_preds == labels).float().mean()
    
    return correct.item()


def calculate_dichotomy_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Вычисление точности для каждой дихотомии отдельно.
    
    Параметры:
        predictions: Предсказания модели (batch_size, 4)
        labels: Истинные метки (batch_size, 4)
        threshold: Порог для бинарной классификации
    
    Возвращает:
        Словарь с точностью для каждой дихотомии
    """
    # Преобразуем вероятности в бинарные предсказания
    binary_preds = (predictions > threshold).float()
    
    accuracies = {}
    dichotomy_names = ['I/E', 'N/S', 'T/F', 'J/P']
    
    for i, name in enumerate(dichotomy_names):
        correct = (binary_preds[:, i] == labels[:, i]).float().mean()
        accuracies[name] = correct.item()
    
    return accuracies


class AverageMeter:
    """Вычисляет и сохраняет среднее и текущее значение."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Сброс всей статистики."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Обновление статистики.
        
        Параметры:
            val: Новое значение
            n: Количество элементов (для усреднения по батчу)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Ранняя остановка для остановки обучения, когда валидационные потери не улучшаются.
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """
        Инициализация ранней остановки.
        
        Параметры:
            patience: Количество эпох ожидания перед остановкой
            min_delta: Минимальное изменение для считаться улучшением
            mode: 'min' для потерь (меньше лучше), 'max' для точности (больше лучше)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = lambda current, best: current < (best - min_delta)
            self.best_score = float('inf')
        else:
            self.monitor_op = lambda current, best: current > (best + min_delta)
            self.best_score = -float('inf')
    
    def __call__(self, score: float) -> bool:
        """
        Проверка, следует ли остановить обучение.
        
        Параметры:
            score: Текущая оценка (потери или точность)
        
        Возвращает:
            True если следует остановить обучение
        """
        if self.monitor_op(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


if __name__ == "__main__":
    # Пример использования
    print("Тестирование метрик MBTI...")
    
    # Создаём тестовые предсказания и метки
    np.random.seed(42)
    n_samples = 100
    
    # Симулируем предсказания (вероятности)
    probabilities = torch.rand(n_samples, 4)
    predictions = (probabilities > 0.5).float()
    
    # Симулируем метки
    labels = torch.randint(0, 2, (n_samples, 4)).float()
    
    # Создаём калькулятор метрик
    metrics_calc = MBTIMetrics()
    
    # Обновляем батчем
    metrics_calc.update(predictions, labels, probabilities)
    
    # Вычисляем метрики
    metrics = metrics_calc.compute()
    
    # Выводим сводку
    metrics_calc.print_summary()
    
    # Тест матриц ошибок
    cm_dict = metrics_calc.get_confusion_matrices()
    print("\nМатрицы ошибок:")
    for i, cm in cm_dict.items():
        print(f"\nДихотомия {i} ({metrics_calc.DICHOTOMY_SHORT[i]}):")
        print(cm)
    
    print("\n✅ Тест метрик пройден!")


"""
Модуль тренировки модели MBTI.

Содержит класс `MBTITrainer`, отвечающий за полный цикл обучения:
- обучение и валидация
- сохранение чекпойнтов
- раннюю остановку
- планировщик скорости обучения
- сбор и вывод метрик
"""
import os
import time
from typing import Dict, Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .metrics import MBTIMetrics, AverageMeter, calculate_dichotomy_accuracy


class MBTITrainer:
    """
    Класс для обучения моделей MBTI.

    Основные возможности:
    • цикл обучения с прогресс-баром
    • валидация после каждой эпохи
    • сохранение лучших и промежуточных чекпойнтов
    • поддержка ранней остановки
    • планировщик LR (ReduceLROnPlateau, StepLR, CosineAnnealing и т. д.)
    • трекинг метрик, включая точность по дихотомиям
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 5,
        gradient_clip: Optional[float] = 1.0,
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
        save_best_only: bool = True
    ):
        """Инициализация тренера.

        Параметры:
            model: обучаемая модель
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            criterion: функция потерь
            optimizer: оптимизатор
            device: устройство ('cuda' / 'cpu')
            scheduler: планировщик LR (опционально)
            early_stopping_patience: терпение ранней остановки
            gradient_clip: значение для обрезки градиентов (None отключает)
            checkpoint_dir: каталог для сохранения моделей
            log_interval: как часто обновлять прогресс-бар (в батчах)
            save_best_only: сохранять только лучшие модели
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_best_only = save_best_only
        
        # Создаём директорию для чекпойнтов
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Состояние обучения
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        self.epochs_no_improve = 0
        self.early_stopping_patience = early_stopping_patience
        
        # История обучения
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        print(f"Тренер инициализирован на устройстве: {device}")
        print(f"В модели {self._count_parameters():,} параметров")
    
    def _count_parameters(self) -> int:
        """Подсчёт обучаемых параметров."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Обучение одной эпохи.
        
        Возвращает:
            Словарь с метриками обучения
        """
        self.model.train()
        
        train_loss = AverageMeter()
        metrics = MBTIMetrics(device=self.device)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Эпоха {self.current_epoch + 1} [Обучение]",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Переносим данные на устройство
            input_ids = batch['input_ids'].to(self.device)
            lengths = batch['length'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Прямой проход
            predictions = self.model(input_ids, lengths)
            loss = self.criterion(predictions, labels)
            
            # Обратный проход
            self.optimizer.zero_grad()
            loss.backward()
            
            # Обрезка градиентов
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            # Обновляем метрики
            batch_size = input_ids.size(0)
            train_loss.update(loss.item(), batch_size)
            
            binary_preds = (predictions > 0.5).float()
            metrics.update(binary_preds, labels, predictions)
            
            # Обновляем прогресс-бар
            if batch_idx % self.log_interval == 0:
                progress_bar.set_postfix({
                    'loss': f'{train_loss.avg:.4f}',
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
            
            self.global_step += 1
        
        # Вычисляем метрики эпохи
        epoch_metrics = metrics.compute()
        epoch_metrics['loss'] = train_loss.avg
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Валидация модели.
        
        Возвращает:
            Словарь с метриками валидации
        """
        self.model.eval()
        
        val_loss = AverageMeter()
        metrics = MBTIMetrics(device=self.device)
        
        progress_bar = tqdm(
            self.val_loader,
            desc=f"Эпоха {self.current_epoch + 1} [Валидация]",
            leave=False
        )
        
        for batch in progress_bar:
            # Переносим данные на устройство
            input_ids = batch['input_ids'].to(self.device)
            lengths = batch['length'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Прямой проход
            predictions = self.model(input_ids, lengths)
            loss = self.criterion(predictions, labels)
            
            # Обновляем метрики
            batch_size = input_ids.size(0)
            val_loss.update(loss.item(), batch_size)
            
            binary_preds = (predictions > 0.5).float()
            metrics.update(binary_preds, labels, predictions)
            
            # Обновляем прогресс-бар
            progress_bar.set_postfix({'loss': f'{val_loss.avg:.4f}'})
        
        # Вычисляем метрики эпохи
        epoch_metrics = metrics.compute()
        epoch_metrics['loss'] = val_loss.avg
        
        return epoch_metrics
    
    def train(
        self,
        num_epochs: int,
        resume_from: Optional[str] = None,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Обучение модели на несколько эпох.
        
        Параметры:
            num_epochs: Количество эпох обучения
            resume_from: Путь к чекпойнту для возобновления
            callback: Опциональная функция обратного вызова после каждой эпохи
        
        Возвращает:
            Историю обучения
        """
        # Возобновляем с чекпойнта если указан
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        start_epoch = self.current_epoch
        print(f"\n{'='*70}")
        print(f"Начинаем обучение с эпохи {start_epoch + 1} до {num_epochs}")
        print(f"{'='*70}\n")
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Обучение одной эпохи
            train_metrics = self.train_epoch()
            
            # Валидация
            val_metrics = self.validate()
            
            # Обновляем скорость обучения
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Обновляем историю
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['overall_accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['overall_accuracy'])
            self.history['learning_rates'].append(current_lr)
            
            # Вычисляем время эпохи
            epoch_time = time.time() - epoch_start_time
            
            # Выводим сводку по эпохе
            self._print_epoch_summary(
                epoch + 1,
                num_epochs,
                train_metrics,
                val_metrics,
                current_lr,
                epoch_time
            )
            
            # Проверяем улучшение
            improved = val_metrics['overall_accuracy'] > self.best_val_accuracy
            
            if improved:
                self.best_val_loss = val_metrics['loss']
                self.best_val_accuracy = val_metrics['overall_accuracy']
                self.epochs_no_improve = 0
                
                # Сохраняем лучшую модель
                self.save_checkpoint(
                    filename='best_model.pth',
                    is_best=True,
                    metrics=val_metrics
                )
                print(f"✓ Новая лучшая модель сохранена! (Val Acc: {self.best_val_accuracy:.4f})")
            else:
                self.epochs_no_improve += 1
            
            # Сохраняем обычный чекпойнт
            if not self.save_best_only or epoch == num_epochs - 1:
                self.save_checkpoint(
                    filename=f'checkpoint_epoch_{epoch + 1}.pth',
                    is_best=False,
                    metrics=val_metrics
                )
            
            # Ранняя остановка
            if self.epochs_no_improve >= self.early_stopping_patience:
                print(f"\n⚠ Сработала ранняя остановка после {epoch + 1} эпох")
                print(f"Нет улучшения в течение {self.early_stopping_patience} эпох")
                break
            
            # Пользовательский обратный вызов
            if callback is not None:
                callback(self, epoch, train_metrics, val_metrics)
        
        print(f"\n{'='*70}")
        print("Обучение завершено!")
        print(f"Лучшая точность на валидации: {self.best_val_accuracy:.4f}")
        print(f"Лучший loss на валидации: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")
        
        return self.history
    
    def _print_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict,
        val_metrics: Dict,
        lr: float,
        epoch_time: float
    ):
        """Вывод сводки для одной эпохи."""
        print(f"\nЭпоха {epoch}/{total_epochs} - {epoch_time:.2f}с")
        print(f"{'─'*70}")
        print(f"Скорость обучения: {lr:.6f}")
        print(f"Обучение Loss: {train_metrics['loss']:.4f} | Обучение Acc: {train_metrics['overall_accuracy']:.4f}")
        print(f"Валидация Loss: {val_metrics['loss']:.4f} | Валидация Acc: {val_metrics['overall_accuracy']:.4f}")
        
        print("\nТочность по дихотомиям:")
        for i, name in enumerate(['I/E', 'N/S', 'T/F', 'J/P']):
            train_acc = train_metrics[f'{name}_accuracy']
            val_acc = val_metrics[f'{name}_accuracy']
            print(f"  {name}: Обуч={train_acc:.4f}, Вал={val_acc:.4f}")
        
        print(f"{'─'*70}")
    
    def save_checkpoint(
        self,
        filename: str = 'checkpoint.pth',
        is_best: bool = False,
        metrics: Optional[Dict] = None
    ):
        """
        Сохранение чекпойнта модели.
        
        Параметры:
            filename: Имя файла чекпойнта
            is_best: Является ли это лучшей моделью
            metrics: Опциональные метрики для сохранения
        """
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'history': self.history,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim_1': self.model.hidden_dim_1,
                'hidden_dim_2': self.model.hidden_dim_2,
                'output_dim': self.model.output_dim,
                'dropout': self.model.dropout_rate,
                'bidirectional': self.model.bidirectional
            }
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if not is_best:
            print(f"Чекпойнт сохранён: {filepath}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Загрузка чекпойнта модели.
        
        Параметры:
            checkpoint_path: Путь к файлу чекпойнта
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.history = checkpoint['history']
        
        print(f"Чекпойнт загружен из: {checkpoint_path}")
        print(f"Продолжаем с эпохи {self.current_epoch}")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_name: str = 'rmsprop',
    learning_rate: float = 0.001,
    scheduler_name: Optional[str] = 'plateau',
    **kwargs
) -> MBTITrainer:
    """
    Фабричная функция для создания тренера с общими настройками.
    
    Параметры:
        model: LSTM модель
        train_loader: Загрузчик обучающих данных
        val_loader: Загрузчик валидационных данных
        optimizer_name: Название оптимизатора ('rmsprop', 'adam', 'adamw', 'sgd')
        learning_rate: Скорость обучения
        scheduler_name: Название планировщика ('plateau', 'step', 'cosine', None)
        **kwargs: Дополнительные аргументы для тренера
    
    Возвращает:
        Настроенный тренер
    """
    # Создаём оптимизатор
    if optimizer_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )
    else:
        raise ValueError(f"Неизвестный оптимизатор: {optimizer_name}")
    
    # Создаём планировщик
    scheduler = None
    if scheduler_name is not None:
        if scheduler_name.lower() == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=3
            )
        elif scheduler_name.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.1
            )
        elif scheduler_name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=50
            )
    
    # Создаём функцию потерь
    if 'weight_tensor' in kwargs and kwargs['weight_tensor'] is not None:
        weight_tensor = kwargs.pop('weight_tensor')
        criterion = nn.BCELoss(weight=weight_tensor)
    else:
        criterion = nn.BCELoss()
    
    # Создаём тренер
    trainer = MBTITrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        **kwargs
    )
    
    return trainer


if __name__ == "__main__":
    print("Модуль тренера успешно загружен!")
    print("Используйте create_trainer() для создания настроенного экземпляра тренера.")


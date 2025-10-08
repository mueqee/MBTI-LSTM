#!/usr/bin/env python3
"""
Умный поиск гиперпараметров для MBTI-LSTM через Optuna.

Вместо тупого перебора используем байесовскую оптимизацию.
Экономим GPU время и находим оптимум быстрее.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
from optuna.trial import Trial
import numpy as np
import pandas as pd

# Добавляем путь к модулям проекта
sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    load_mbti_data,
    create_data_loaders,
    MBTIPostPreprocessor
)
from src.models import LSTMMBTIClassifier
from src.training import MBTITrainer, MBTIMetrics


def objective(trial: Trial, args: argparse.Namespace) -> float:
    """
    Целевая функция для Optuna.
    
    Возвращает отрицательную val_accuracy (минимизируем).
    """
    
    # Гиперпараметры для поиска
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'hidden_dim_1': trial.suggest_categorical('hidden_dim_1', [64, 128, 256]),
        'hidden_dim_2': trial.suggest_categorical('hidden_dim_2', [32, 64, 128]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'rmsprop']),
        'weight_decay': trial.suggest_float('weight_decay', 0, 0.01, step=0.001),
    }
    
    # Зависимые параметры
    if hyperparams['optimizer'] == 'adam':
        hyperparams['beta1'] = trial.suggest_float('adam_beta1', 0.8, 0.95)
        hyperparams['beta2'] = trial.suggest_float('adam_beta2', 0.99, 0.999)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: Тестируем конфигурацию")
    print(f"{'='*60}")
    for k, v in hyperparams.items():
        print(f"  {k}: {v}")
    
    try:
        # Загружаем данные
        print("\nЗагружаем данные...")
        df = pd.read_csv(args.data_path)
        
        # Создаём препроцессор
        preprocessor = MBTIPostPreprocessor(
            lowercase=True,
            remove_urls=True,
            remove_special_chars=True,
            remove_stopwords=True,
            lemmatize=True
        )
        
        # Создаём DataLoader'ы с новым batch_size
        loaders = create_data_loaders(
            df,
            preprocessor=preprocessor,
            max_length=300,
            batch_size=hyperparams['batch_size'],
            val_split=0.15,
            test_split=0.15,
            num_workers=2
        )

        train_loader = loaders['train']
        val_loader = loaders['val']
        vocab = loaders['vocabulary']
        
        print(f"Размер словаря: {len(vocab)}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Создаём модель с новыми параметрами
        model = LSTMMBTIClassifier(
            vocab_size=len(vocab),
            embedding_dim=300,
            hidden_dim_1=hyperparams['hidden_dim_1'],
            hidden_dim_2=hyperparams['hidden_dim_2'],
            dropout=hyperparams['dropout']
        )
        
        # Настраиваем оптимизатор
        if hyperparams['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=hyperparams['learning_rate'],
                betas=(hyperparams['beta1'], hyperparams['beta2']),
                weight_decay=hyperparams['weight_decay']
            )
        elif hyperparams['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=hyperparams['learning_rate'],
                weight_decay=hyperparams['weight_decay']
            )
        else:  # rmsprop
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=hyperparams['learning_rate'],
                alpha=0.99,
                weight_decay=hyperparams['weight_decay']
            )
        
        # Создаём trainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nИспользуем устройство: {device}")
        
        criterion = nn.BCELoss()
        
        trainer = MBTITrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            checkpoint_dir='experiments/tmp_optuna',
            save_best_only=False,
            early_stopping_patience=2
        )
        
        # Обучаем модель (мало эпох для быстрого поиска)
        print(f"\nОбучаем {args.search_epochs} эпох...")
        best_val_acc = 0
        
        for epoch in range(1, args.search_epochs + 1):
            # Обучение
            train_loss, train_acc = trainer.train_epoch()
            
            # Валидация
            val_loss, val_acc, val_dichotomies = trainer.validate()
            
            print(f"Эпоха {epoch}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
            
            # Обновляем лучший результат
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Pruning - останавливаем плохие попытки раньше
            trial.report(val_acc, epoch)
            if trial.should_prune():
                print("Пробуем прунинг - результаты не улучшаются")
                raise optuna.TrialPruned()
        
        print(f"\nЛучшая точность: {best_val_acc:.4f}")
        return -best_val_acc  # Минимизируем отрицательную accuracy
        
    except Exception as e:
        print(f"Ошибка в trial {trial.number}: {e}")
        return 0.0  # Возвращаем худший результат при ошибке


def main():
    parser = argparse.ArgumentParser(description="Умный поиск гиперпараметров для MBTI-LSTM")
    parser.add_argument('--data_path', type=str, default='data/raw/mbti_dataset.csv',
                        help='Путь к датасету')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Количество попыток Optuna')
    parser.add_argument('--search_epochs', type=int, default=5,
                        help='Количество эпох для каждой попытки')
    parser.add_argument('--max_length', type=int, default=500,
                        help='Максимальная длина последовательности')
    parser.add_argument('--study_name', type=str, default='mbti_lstm_optimization',
                        help='Название исследования Optuna')
    parser.add_argument('--storage', type=str, default='sqlite:///experiments/optuna.db',
                        help='База данных для Optuna')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Таймаут в секундах')
    
    args = parser.parse_args()
    
    # Создаём директорию для экспериментов
    Path('experiments').mkdir(exist_ok=True)
    
    print("="*60)
    print("УМНЫЙ ПОИСК ГИПЕРПАРАМЕТРОВ ДЛЯ MBTI-LSTM")
    print("="*60)
    print(f"Датасет: {args.data_path}")
    print(f"Количество попыток: {args.n_trials}")
    print(f"Эпох на попытку: {args.search_epochs}")
    print(f"Study: {args.study_name}")
    print("="*60)
    
    # Создаём или загружаем study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
            interval_steps=1
        )
    )
    
    # Запускаем оптимизацию
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=True
    )
    
    # Выводим результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ПОИСКА")
    print("="*60)
    
    print(f"\nЛучший результат: {-study.best_value:.4f} accuracy")
    print("\nЛучшие гиперпараметры:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Сохраняем лучшую конфигурацию
    best_config_path = Path('experiments') / f'best_config_{datetime.now():%Y%m%d_%H%M%S}.json'
    with open(best_config_path, 'w') as f:
        json.dump({
            'accuracy': -study.best_value,
            'params': study.best_params,
            'n_trials': len(study.trials),
            'datetime': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nЛучшая конфигурация сохранена: {best_config_path}")
    
    # Статистика по попыткам
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print(f"\nСтатистика:")
    print(f"  Завершено: {len(completed_trials)}")
    print(f"  Прунинг: {len(pruned_trials)}")
    print(f"  Среднее время на trial: {np.mean([t.duration.total_seconds() for t in completed_trials if t.duration]):.1f} сек")
    
    # Топ-5 конфигураций
    print("\nТоп-5 конфигураций:")
    trials_df = study.trials_dataframe()
    top_trials = trials_df.nsmallest(5, 'value')
    for idx, row in top_trials.iterrows():
        print(f"  {idx+1}. Accuracy: {-row['value']:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Скрипт обучения MBTI-LSTM модели

Этот скрипт предоставляет интерфейс командной строки для обучения LSTM моделей
для классификации типов личности MBTI.

Использование:
    python scripts/train.py --data_path data/raw/mbti_dataset.csv --num_epochs 50
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch

# Добавляем корень проекта в path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessor import create_preprocessor
from src.data.dataset import load_mbti_data, create_data_loaders
from src.models.lstm_model import create_model
from src.training.trainer import create_trainer


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Обучение LSTM модели для классификации MBTI"
    )
    
    # Аргументы данных
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Путь к CSV файлу датасета MBTI"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/model_config.yaml",
        help="Путь к файлу конфигурации"
    )
    
    # Аргументы модели
    parser.add_argument(
        "--model_type",
        type=str,
        default="lstm",
        choices=["lstm", "lstm_attention"],
        help="Тип модели для обучения"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=300,
        help="Размерность эмбеддингов"
    )
    parser.add_argument(
        "--hidden_dim_1",
        type=int,
        default=128,
        help="Скрытая размерность первого LSTM слоя"
    )
    parser.add_argument(
        "--hidden_dim_2",
        type=int,
        default=64,
        help="Скрытая размерность второго LSTM слоя"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Коэффициент dropout"
    )
    
    # Аргументы обучения
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Количество эпох обучения"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Размер батча"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Скорость обучения"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="rmsprop",
        choices=["rmsprop", "adam", "adamw", "sgd"],
        help="Оптимизатор"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "step", "cosine", "none"],
        help="Планировщик скорости обучения"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Терпение для ранней остановки"
    )
    
    # Аргументы обработки данных
    parser.add_argument(
        "--max_length",
        type=int,
        default=500,
        help="Максимальная длина последовательности"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Доля валидационной выборки"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Доля тестовой выборки"
    )
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Балансировка классов в обучающей выборке"
    )
    
    # Другие аргументы
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Каталог для сохранения чекпойнтов"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Путь к чекпойнту для продолжения"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "auto"],
        help="Устройство для обучения"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Начальное состояние генератора случайных чисел"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Количество воркеров для загрузки данных"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Загрузка конфигурации из YAML файла."""
    if not os.path.exists(config_path):
        print(f"Предупреждение: Файл конфигурации {config_path} не найден. Используются значения по умолчанию.")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def set_seed(seed: int):
    """Установка начального состояния для воспроизводимости."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Основная функция обучения."""
    # Парсим аргументы
    args = parse_args()
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    
    # Устанавливаем seed
    set_seed(args.seed)
    
    # Определяем устройство
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"\n{'='*70}")
    print("ОБУЧЕНИЕ MBTI-LSTM")
    print(f"{'='*70}")
    print(f"Устройство: {device}")
    print(f"Данные: {args.data_path}")
    print(f"Модель: {args.model_type}")
    print(f"Эпохи: {args.num_epochs}")
    print(f"Размер батча: {args.batch_size}")
    print(f"Скорость обучения: {args.learning_rate}")
    print(f"Оптимизатор: {args.optimizer}")
    print(f"{'='*70}\n")
    
    # Загружаем данные
    print("Загрузка данных...")
    data = load_mbti_data(args.data_path)
    
    # Создаём препроцессор
    print("Создание препроцессора...")
    preprocessor = create_preprocessor("mbti")
    
    # Создаём загрузчики данных
    print("Создание загрузчиков данных...")
    data_loaders = create_data_loaders(
        data=data,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        max_length=args.max_length,
        balance_classes=args.balance_classes,
        random_state=args.seed,
        num_workers=min(2, args.num_workers)  # Ограничение для Colab
    )
    
    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    vocabulary = data_loaders['vocabulary']
    
    print(f"Размер словаря: {len(vocabulary)}")
    print(f"Обучающих сэмплов: {len(train_loader.dataset)}")
    print(f"Валидационных сэмплов: {len(val_loader.dataset)}")
    print(f"Тестовых сэмплов: {len(test_loader.dataset)}")
    
    # Создаём модель
    print(f"\nСоздание {args.model_type} модели...")
    model = create_model(
        vocab_size=len(vocabulary),
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
        dropout=args.dropout,
        bidirectional=True
    )
    
    print(model.get_architecture_summary())
    
    # Создаём тренер
    print("Создание тренера...")
    scheduler_name = None if args.scheduler == "none" else args.scheduler
    
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        scheduler_name=scheduler_name,
        device=device,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Обучаем модель
    print("\nНачало обучения...")
    history = trainer.train(
        num_epochs=args.num_epochs,
        resume_from=args.resume_from
    )
    
    # Оценка на тестовой выборке
    print("\nОценка на тестовой выборке...")
    from src.training.metrics import MBTIMetrics
    
    model.eval()
    test_metrics = MBTIMetrics(device=device)
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            lengths = batch['length'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(input_ids, lengths)
            binary_preds = (predictions > 0.5).float()
            
            test_metrics.update(binary_preds, labels, predictions)
    
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
    test_metrics.print_summary()
    
    # Сохраняем словарь
    vocab_path = os.path.join(args.checkpoint_dir, "vocabulary.pth")
    vocabulary.save(vocab_path)
    print(f"Словарь сохранён в: {vocab_path}")
    
    # Сохраняем финальную модель
    final_model_path = os.path.join(args.checkpoint_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': len(vocabulary),
            'model_type': args.model_type,
            'embedding_dim': args.embedding_dim,
            'hidden_dim_1': args.hidden_dim_1,
            'hidden_dim_2': args.hidden_dim_2,
            'dropout': args.dropout,
            'bidirectional': True
        },
        'test_metrics': test_metrics.compute()
    }, final_model_path)
    
    print(f"Финальная модель сохранена в: {final_model_path}")
    
    print("\n" + "="*70)
    print("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()


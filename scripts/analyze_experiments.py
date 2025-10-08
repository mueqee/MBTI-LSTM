#!/usr/bin/env python3
"""
Анализ результатов экспериментов с гиперпараметрами.

Визуализация и сравнение разных конфигураций.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import optuna
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)


def load_best_configs(experiments_dir: Path) -> List[Dict[str, Any]]:
    """Загружает все сохранённые конфигурации."""
    configs = []
    for config_file in experiments_dir.glob('best_config_*.json'):
        with open(config_file, 'r') as f:
            config = json.load(f)
            config['filename'] = config_file.name
            configs.append(config)
    return sorted(configs, key=lambda x: x['accuracy'], reverse=True)


def print_comparison_table(configs: List[Dict[str, Any]]):
    """Выводит таблицу сравнения конфигураций."""
    print("\n" + "="*80)
    print("СРАВНЕНИЕ ЛУЧШИХ КОНФИГУРАЦИЙ")
    print("="*80)
    
    if not configs:
        print("Нет сохранённых конфигураций")
        return
    
    # Собираем все уникальные параметры
    all_params = set()
    for config in configs:
        all_params.update(config['params'].keys())
    
    # Создаём DataFrame для красивого вывода
    data = []
    for i, config in enumerate(configs[:5], 1):  # Топ-5
        row = {
            'Rank': i,
            'Accuracy': f"{config['accuracy']:.4f}",
            'Trials': config.get('n_trials', 'N/A')
        }
        for param in sorted(all_params):
            value = config['params'].get(param, 'N/A')
            if isinstance(value, float):
                row[param] = f"{value:.4f}"
            else:
                row[param] = str(value)
        data.append(row)
    
    df = pd.DataFrame(data)
    print(df.to_string(index=False))
    
    # Лучшая конфигурация детально
    best = configs[0]
    print("\n" + "="*80)
    print("ЛУЧШАЯ КОНФИГУРАЦИЯ")
    print("="*80)
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Файл: {best['filename']}")
    print(f"Дата: {best.get('datetime', 'N/A')}")
    print("\nПараметры:")
    for key, value in sorted(best['params'].items()):
        print(f"  {key}: {value}")


def visualize_study(study_name: str, storage: str):
    """Создаёт визуализации для Optuna study."""
    print("\n" + "="*80)
    print("ВИЗУАЛИЗАЦИЯ ЭКСПЕРИМЕНТОВ")
    print("="*80)
    
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        
        print(f"Загружено исследование: {study_name}")
        print(f"Всего попыток: {len(study.trials)}")
        
        # Создаём директорию для графиков
        plots_dir = Path('experiments/plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. История оптимизации
        fig = plot_optimization_history(study)
        fig.write_html(plots_dir / 'optimization_history.html')
        print(f"  ✓ История оптимизации: {plots_dir}/optimization_history.html")
        
        # 2. Важность параметров
        fig = plot_param_importances(study)
        fig.write_html(plots_dir / 'param_importances.html')
        print(f"  ✓ Важность параметров: {plots_dir}/param_importances.html")
        
        # 3. Параллельные координаты
        fig = plot_parallel_coordinate(study)
        fig.write_html(plots_dir / 'parallel_coordinate.html')
        print(f"  ✓ Параллельные координаты: {plots_dir}/parallel_coordinate.html")
        
        # 4. Срезы параметров
        fig = plot_slice(study)
        fig.write_html(plots_dir / 'param_slices.html')
        print(f"  ✓ Срезы параметров: {plots_dir}/param_slices.html")
        
        # 5. Статистика по оптимизаторам (если есть)
        trials_df = study.trials_dataframe()
        if 'params_optimizer' in trials_df.columns:
            plt.figure(figsize=(10, 6))
            optimizer_stats = trials_df.groupby('params_optimizer')['value'].agg(['mean', 'std', 'min'])
            optimizer_stats['mean'] = -optimizer_stats['mean']  # Конвертируем обратно в accuracy
            optimizer_stats['min'] = -optimizer_stats['min']
            
            ax = optimizer_stats[['mean']].plot(kind='bar', yerr=optimizer_stats['std'])
            ax.set_ylabel('Accuracy')
            ax.set_title('Сравнение оптимизаторов')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / 'optimizer_comparison.png')
            print(f"  ✓ Сравнение оптимизаторов: {plots_dir}/optimizer_comparison.png")
        
        print("\n✅ Все визуализации сохранены в experiments/plots/")
        
    except Exception as e:
        print(f"Ошибка при загрузке study: {e}")


def suggest_next_experiments(configs: List[Dict[str, Any]]):
    """Предлагает следующие эксперименты на основе результатов."""
    print("\n" + "="*80)
    print("РЕКОМЕНДАЦИИ ДЛЯ СЛЕДУЮЩИХ ЭКСПЕРИМЕНТОВ")
    print("="*80)
    
    if not configs:
        print("Недостаточно данных для рекомендаций")
        return
    
    best = configs[0]
    best_params = best['params']
    
    print(f"На основе лучшего результата ({best['accuracy']:.4f}):")
    
    # Анализируем параметры
    recommendations = []
    
    # Learning rate
    if 'learning_rate' in best_params:
        lr = best_params['learning_rate']
        if lr < 0.001:
            recommendations.append(f"• Learning rate низкий ({lr:.4f}). Попробуй warmup или cosine annealing")
        elif lr > 0.005:
            recommendations.append(f"• Learning rate высокий ({lr:.4f}). Добавь gradient clipping")
    
    # Dropout
    if 'dropout' in best_params:
        dropout = best_params['dropout']
        if dropout < 0.2:
            recommendations.append(f"• Dropout низкий ({dropout}). Может быть переобучение")
        elif dropout > 0.4:
            recommendations.append(f"• Dropout высокий ({dropout}). Модель может недоучиваться")
    
    # Hidden dimensions
    if 'hidden_dim_1' in best_params and 'hidden_dim_2' in best_params:
        h1, h2 = best_params['hidden_dim_1'], best_params['hidden_dim_2']
        if h1 / h2 < 1.5:
            recommendations.append(f"• Размеры слоёв близки ({h1}/{h2}). Попробуй больший контраст")
    
    # Оптимизатор
    if 'optimizer' in best_params:
        opt = best_params['optimizer']
        if opt == 'rmsprop':
            recommendations.append("• RMSprop хорошо работает. Но попробуй AdamW с weight_decay")
        elif opt == 'adam':
            recommendations.append("• Adam склонен к переобучению. Добавь L2 регуляризацию")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("• Конфигурация выглядит оптимальной!")
    
    print("\nСледующие шаги:")
    print("1. Запусти финальное обучение с лучшими параметрами на 50 эпох")
    print("2. Добавь attention механизм к этой конфигурации")
    print("3. Попробуй pretrained embeddings (FastText для английского)")


def main():
    parser = argparse.ArgumentParser(description="Анализ результатов экспериментов")
    parser.add_argument('--experiments_dir', type=str, default='experiments',
                        help='Директория с экспериментами')
    parser.add_argument('--study_name', type=str, default='mbti_lstm_optimization',
                        help='Название Optuna study')
    parser.add_argument('--storage', type=str, default='sqlite:///experiments/optuna.db',
                        help='База данных Optuna')
    
    args = parser.parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    
    # Загружаем конфигурации
    configs = load_best_configs(experiments_dir)
    
    # Выводим таблицу сравнения
    print_comparison_table(configs)
    
    # Визуализация
    if (experiments_dir / 'optuna.db').exists():
        visualize_study(args.study_name, args.storage)
    
    # Рекомендации
    suggest_next_experiments(configs)


if __name__ == "__main__":
    main()

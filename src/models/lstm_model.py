"""
LSTM модель для классификации типов личности MBTI

Этот модуль реализует LSTM архитектуру из магистерской диссертации:
- Embedding слой (300d)
- BiLSTM слой 1 (128 нейронов)
- BiLSTM слой 2 (64 нейрона)
- Dropout (0.2)
- Полносвязный слой (64 нейрона, ReLU)
- Выходной слой (4 нейрона для 4 дихотомий, Sigmoid)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMMBTIClassifier(nn.Module):
    """
    LSTM-классификатор для типов личности MBTI.
    
    Модель предсказывает 4 независимые бинарные классификации (дихотомии):
    - I/E (Интроверсия/Экстраверсия)
    - N/S (Интуиция/Сенсорика)
    - T/F (Мышление/Чувствование)
    - J/P (Суждение/Восприятие)
    
    Архитектура:
        Embedding → BiLSTM(128) → BiLSTM(64) → Dropout → FC(64) → Output(4)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        output_dim: int = 4,
        dropout: float = 0.2,
        num_layers_1: int = 1,
        num_layers_2: int = 1,
        bidirectional: bool = True,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        padding_idx: int = 0,
    ):
        """
        Инициализация LSTM MBTI классификатора.
        
        Параметры:
            vocab_size: Размер словаря
            embedding_dim: Размерность эмбеддингов слов (по умолчанию: 300)
            hidden_dim_1: Скрытая размерность первого LSTM слоя (по умолчанию: 128)
            hidden_dim_2: Скрытая размерность второго LSTM слоя (по умолчанию: 64)
            output_dim: Количество выходных классов (4 дихотомии)
            dropout: Коэффициент dropout (по умолчанию: 0.2)
            num_layers_1: Количество LSTM слоёв в первом блоке (по умолчанию: 1)
            num_layers_2: Количество LSTM слоёв во втором блоке (по умолчанию: 1)
            bidirectional: Использовать ли двунаправленный LSTM (по умолчанию: True)
            pretrained_embeddings: Опциональные предобученные веса эмбеддингов
            freeze_embeddings: Замораживать ли веса эмбеддингов
            padding_idx: Индекс токена для padding
        """
        super(LSTMMBTIClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Слой эмбеддингов
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        # Загружаем предобученные эмбеддинги если они предоставлены
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            
        # Замораживаем эмбеддинги если указано
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # Первый BiLSTM слой (128 нейронов)
        self.lstm1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim_1,
            num_layers=num_layers_1,
            batch_first=True,
            dropout=dropout if num_layers_1 > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Второй BiLSTM слой (64 нейрона)
        lstm1_output_dim = hidden_dim_1 * self.num_directions
        self.lstm2 = nn.LSTM(
            input_size=lstm1_output_dim,
            hidden_size=hidden_dim_2,
            num_layers=num_layers_2,
            batch_first=True,
            dropout=dropout if num_layers_2 > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Слой dropout
        self.dropout = nn.Dropout(dropout)
        
        # Полносвязный слой (64 нейрона, ReLU)
        lstm2_output_dim = hidden_dim_2 * self.num_directions
        self.fc1 = nn.Linear(lstm2_output_dim, 64)
        self.relu = nn.ReLU()
        
        # Выходной слой (4 дихотомии, Sigmoid)
        self.fc2 = nn.Linear(64, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Инициализируем веса
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов с использованием Xavier/Glorot."""
        for name, param in self.named_parameters():
            if 'embedding' in name:
                continue  # Пропускаем веса эмбеддингов
            elif 'weight' in name:
                if 'lstm' in name:
                    # Инициализация весов LSTM
                    nn.init.xavier_uniform_(param)
                else:
                    # Линейные слои
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прямой проход через сеть.
        
        Параметры:
            input_ids: ID входных токенов, форма (batch_size, seq_len)
            lengths: Фактические длины последовательностей (без padding), форма (batch_size,)
        
        Возвращает:
            Предсказания для 4 дихотомий, форма (batch_size, 4)
        """
        # Эмбеддинг: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # Упаковываем padded последовательности если предоставлены длины
        if lengths is not None:
            # Сортировка по длине (требуется для pack_padded_sequence)
            lengths_sorted, sorted_idx = lengths.sort(descending=True)
            embedded_sorted = embedded[sorted_idx]
            
            # Упаковываем последовательности
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded_sorted,
                lengths_sorted.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            
            # Первый LSTM
            packed_output1, (hidden1, cell1) = self.lstm1(packed_embedded)
            
            # Распаковываем последовательности
            output1, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output1,
                batch_first=True
            )
            
            # Снова упаковываем для второго LSTM
            packed_output1 = nn.utils.rnn.pack_padded_sequence(
                output1,
                lengths_sorted.cpu(),
                batch_first=True,
                enforce_sorted=True
            )
            
            # Второй LSTM
            packed_output2, (hidden2, cell2) = self.lstm2(packed_output1)
            
            # Распаковываем последовательности
            output2, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output2,
                batch_first=True
            )
            
            # Восстанавливаем исходный порядок
            _, unsorted_idx = sorted_idx.sort()
            output2 = output2[unsorted_idx]
            hidden2 = hidden2[:, unsorted_idx, :]
            
        else:
            # Без упаковки (проще, но менее эффективно)
            output1, (hidden1, cell1) = self.lstm1(embedded)
            output2, (hidden2, cell2) = self.lstm2(output1)
        
        # Используем финальное скрытое состояние из последнего слоя
        # Для двунаправленного LSTM: объединяем прямое и обратное скрытые состояния
        if self.bidirectional:
            # hidden2 форма: (num_layers * 2, batch_size, hidden_dim_2)
            hidden_forward = hidden2[-2, :, :]  # Прямое направление
            hidden_backward = hidden2[-1, :, :]  # Обратное направление
            hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            hidden_concat = hidden2[-1, :, :]
        
        # Dropout
        hidden_dropped = self.dropout(hidden_concat)
        
        # Полносвязный слой с ReLU
        fc1_out = self.fc1(hidden_dropped)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        # Выходной слой с Sigmoid
        output = self.fc2(fc1_out)
        output = self.sigmoid(output)
        
        return output
    
    def predict(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Сделать предсказания (бинарные классификации для каждой дихотомии).
        
        Параметры:
            input_ids: ID входных токенов
            lengths: Длины последовательностей
        
        Возвращает:
            Бинарные предсказания (0 или 1) для каждой дихотомии
        """
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(input_ids, lengths)
            predictions = (probabilities > 0.5).long()
        return predictions
    
    def get_mbti_type(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> list:
        """
        Получить строки MBTI типов из входных данных.
        
        Параметры:
            input_ids: ID входных токенов
            lengths: Длины последовательностей
        
        Возвращает:
            Список строк MBTI типов (напр., ['INTJ', 'ESFP', ...])
        """
        predictions = self.predict(input_ids, lengths)
        
        # Отображение дихотомий
        dichotomies = [
            ['I', 'E'],  # Интроверсия/Экстраверсия
            ['N', 'S'],  # Интуиция/Сенсорика
            ['T', 'F'],  # Мышление/Чувствование
            ['J', 'P']   # Суждение/Восприятие
        ]
        
        mbti_types = []
        for pred in predictions:
            mbti_type = ''.join([dichotomies[i][pred[i].item()] for i in range(4)])
            mbti_types.append(mbti_type)
        
        return mbti_types
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Подсчёт общего и обучаемого количества параметров.
        
        Возвращает:
            Кортеж (всего_параметров, обучаемых_параметров)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def get_architecture_summary(self) -> str:
        """
        Получить сводку по архитектуре модели.
        
        Возвращает:
            Строка с описанием архитектуры модели
        """
        total_params, trainable_params = self.count_parameters()
        
        summary = f"""
Архитектура LSTM MBTI классификатора
{'=' * 50}
Слой эмбеддингов:
  - Размер словаря: {self.vocab_size:,}
  - Размерность эмбеддинга: {self.embedding_dim}
  
LSTM слой 1:
  - Скрытая размерность: {self.hidden_dim_1}
  - Двунаправленный: {self.bidirectional}
  - Выходная размерность: {self.hidden_dim_1 * self.num_directions}
  
LSTM слой 2:
  - Скрытая размерность: {self.hidden_dim_2}
  - Двунаправленный: {self.bidirectional}
  - Выходная размерность: {self.hidden_dim_2 * self.num_directions}
  
Dropout: {self.dropout_rate}

Полносвязные слои:
  - FC1: {self.hidden_dim_2 * self.num_directions} -> 64 (ReLU)
  - FC2: 64 -> {self.output_dim} (Sigmoid)

Параметры:
  - Всего: {total_params:,}
  - Обучаемых: {trainable_params:,}
{'=' * 50}
"""
        return summary


class LSTMWithAttention(LSTMMBTIClassifier):
    """
    LSTM модель с механизмом внимания для классификации MBTI.
    
    Расширение базовой LSTM модели, которое добавляет слой внимания для 
    фокусировки на важных частях входной последовательности.
    """
    
    def __init__(self, *args, **kwargs):
        """Инициализация LSTM с механизмом внимания."""
        super(LSTMWithAttention, self).__init__(*args, **kwargs)
        
        # Слой внимания
        lstm2_output_dim = self.hidden_dim_2 * self.num_directions
        self.attention = nn.Linear(lstm2_output_dim, 1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Прямой проход с механизмом внимания.
        
        Параметры:
            input_ids: ID входных токенов
            lengths: Длины последовательностей
        
        Возвращает:
            Предсказания для 4 дихотомий
        """
        # Эмбеддинг
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)
        
        # LSTM слои
        output1, _ = self.lstm1(embedded)
        output2, _ = self.lstm2(output1)
        
        # Механизм внимания
        # output2 форма: (batch_size, seq_len, hidden_dim_2 * num_directions)
        attention_weights = self.attention(output2)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Применяем внимание
        attended_output = torch.sum(output2 * attention_weights, dim=1)  # (batch_size, hidden_dim_2 * num_directions)
        
        # Dropout
        attended_output = self.dropout(attended_output)
        
        # Полносвязные слои
        fc1_out = self.fc1(attended_output)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        # Выход
        output = self.fc2(fc1_out)
        output = self.sigmoid(output)
        
        return output


def create_model(
    vocab_size: int,
    model_type: str = "lstm",
    **kwargs
) -> nn.Module:
    """
    Фабричная функция для создания LSTM моделей.
    
    Параметры:
        vocab_size: Размер словаря
        model_type: Тип модели ('lstm' или 'lstm_attention')
        **kwargs: Дополнительные аргументы для конструктора модели
    
    Возвращает:
        Экземпляр LSTM модели
    """
    if model_type == "lstm":
        return LSTMMBTIClassifier(vocab_size, **kwargs)
    elif model_type == "lstm_attention":
        return LSTMWithAttention(vocab_size, **kwargs)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")


if __name__ == "__main__":
    # Пример использования и тестирование
    print("Тестирование LSTM MBTI классификатора...")
    
    # Создаём тестовую модель
    vocab_size = 10000
    model = LSTMMBTIClassifier(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim_1=128,
        hidden_dim_2=64,
        dropout=0.2,
        bidirectional=True
    )
    
    print(model.get_architecture_summary())
    
    # Тест прямого прохода
    batch_size = 8
    seq_len = 100
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.randint(50, seq_len, (batch_size,))
    
    output = model(input_ids, lengths)
    print(f"\nФорма входа: {input_ids.shape}")
    print(f"Форма выхода: {output.shape}")
    print(f"Выход (вероятности):\n{output}")
    
    # Тест предсказаний
    mbti_types = model.get_mbti_type(input_ids, lengths)
    print(f"\nПредсказанные MBTI типы: {mbti_types}")
    
    print("\n✅ Тест модели пройден!")


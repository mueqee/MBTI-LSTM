"""
Модуль предобработки текста для классификации MBTI

Этот модуль предоставляет утилиты для предобработки текста 
для классификации типов личности MBTI из постов соцсетей.
"""

import re
import string
from typing import List, Optional, Union
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy


class TextPreprocessor:
    """
    Препроцессор текста для классификации MBTI.
    
    Обрабатывает:
    - Удаление URL
    - Удаление специальных символов
    - Токенизация
    - Удаление стоп-слов
    - Лемматизация
    - Приведение к нижнему регистру
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_special_chars: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        language: str = "english",
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm",
        min_token_length: int = 2,
        max_token_length: int = 50,
    ):
        """
        Инициализация препроцессора текста.
        
        Параметры:
            lowercase: Преобразовать текст в нижний регистр
            remove_urls: Удалять URL из текста
            remove_special_chars: Удалять специальные символы
            remove_numbers: Удалять числа
            remove_stopwords: Удалять стоп-слова
            lemmatize: Применять лемматизацию
            language: Язык для стоп-слов (по умолчанию: 'english')
            use_spacy: Использовать spaCy вместо NLTK (точнее, но медленнее)
            spacy_model: Название модели spaCy
            min_token_length: Минимальная длина токена для сохранения
            max_token_length: Максимальная длина токена для сохранения
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_special_chars = remove_special_chars
        self.remove_numbers = remove_numbers
        self.remove_stopwords_flag = remove_stopwords
        self.lemmatize_flag = lemmatize
        self.language = language
        self.use_spacy = use_spacy
        self.min_token_length = min_token_length
        self.max_token_length = max_token_length
        
        # Инициализируем ресурсы NLTK
        if not use_spacy:
            self._download_nltk_resources()
            self.stopwords = set(stopwords.words(language))
            self.lemmatizer = WordNetLemmatizer()
        else:
            # Инициализируем spaCy
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"Модель spaCy '{spacy_model}' не найдена. Загружаю...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", spacy_model])
                self.nlp = spacy.load(spacy_model)
            
            self.stopwords = self.nlp.Defaults.stop_words
        
        # Компилируем regex паттерны для эффективности
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.number_pattern = re.compile(r'\d+')
        
    def _download_nltk_resources(self):
        """Загрузка требуемых ресурсов NLTK если они недоступны."""
        resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
    
    def clean_text(self, text: str) -> str:
        """
        Очистка текста от URL, упоминаний, хэштегов и т.д.
        
        Параметры:
            text: Входной текст
        
        Возвращает:
            Очищенный текст
        """
        if not isinstance(text, str):
            return ""
        
        # Нормализуем unicode символы
        text = unicodedata.normalize('NFKD', text)
        
        # Удаляем URL
        if self.remove_urls:
            text = self.url_pattern.sub('', text)
        
        # Удаляем упоминания и хэштеги (но оставляем текст после #)
        text = self.mention_pattern.sub('', text)
        text = self.hashtag_pattern.sub(lambda m: m.group(0)[1:], text)  # Удаляем # но оставляем слово
        
        # Удаляем лишние пробелы
        text = ' '.join(text.split())
        
        return text
    
    def remove_special_characters(self, text: str) -> str:
        """
        Удаление специальных символов и пунктуации.
        
        Параметры:
            text: Входной текст
        
        Возвращает:
            Текст без специальных символов
        """
        if self.remove_numbers:
            text = self.number_pattern.sub('', text)
        
        # Удаляем пунктуацию, но оставляем апострофы для сокращений
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Удаляем лишние пробелы
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Токенизация текста на слова.
        
        Параметры:
            text: Входной текст
        
        Возвращает:
            Список токенов
        """
        if self.use_spacy:
            doc = self.nlp(text)
            tokens = [token.text for token in doc]
        else:
            tokens = word_tokenize(text)
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Удаление стоп-слов из списка токенов.
        
        Параметры:
            tokens: Список токенов
        
        Возвращает:
            Отфильтрованные токены
        """
        return [token for token in tokens if token.lower() not in self.stopwords]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Лемматизация токенов.
        
        Параметры:
            tokens: Список токенов
        
        Возвращает:
            Лемматизированные токены
        """
        if self.use_spacy:
            text = ' '.join(tokens)
            doc = self.nlp(text)
            return [token.lemma_ for token in doc]
        else:
            return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def filter_tokens(self, tokens: List[str]) -> List[str]:
        """
        Фильтрация токенов по длине и валидности.
        
        Параметры:
            tokens: Список токенов
        
        Возвращает:
            Отфильтрованные токены
        """
        filtered = []
        for token in tokens:
            # Проверяем длину
            if len(token) < self.min_token_length or len(token) > self.max_token_length:
                continue
            # Оставляем только алфавитные токены
            if token.isalpha():
                filtered.append(token)
        
        return filtered
    
    def preprocess(self, text: str) -> List[str]:
        """
        Полный пайплайн предобработки.
        
        Параметры:
            text: Входной текст
        
        Возвращает:
            Список предобработанных токенов
        """
        # Очищаем текст
        text = self.clean_text(text)
        
        # Приводим к нижнему регистру
        if self.lowercase:
            text = text.lower()
        
        # Удаляем специальные символы
        if self.remove_special_chars:
            text = self.remove_special_characters(text)
        
        # Токенизируем
        tokens = self.tokenize(text)
        
        # Фильтруем токены
        tokens = self.filter_tokens(tokens)
        
        # Удаляем стоп-слова
        if self.remove_stopwords_flag:
            tokens = self.remove_stopwords(tokens)
        
        # Лемматизируем
        if self.lemmatize_flag:
            tokens = self.lemmatize(tokens)
        
        return tokens
    
    def preprocess_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Предобработка батча текстов.
        
        Параметры:
            texts: Список входных текстов
        
        Возвращает:
            Список списков предобработанных токенов
        """
        return [self.preprocess(text) for text in texts]
    
    def preprocess_to_string(self, text: str) -> str:
        """
        Предобработка текста с возвратом строки вместо списка токенов.
        
        Параметры:
            text: Входной текст
        
        Возвращает:
            Предобработанный текст как строка
        """
        tokens = self.preprocess(text)
        return ' '.join(tokens)


class MBTIPostPreprocessor(TextPreprocessor):
    """
    Специализированный препроцессор для постов MBTI из соцсетей.
    
    Обрабатывает MBTI-специфичные паттерны и множественные посты на пользователя.
    """
    
    def __init__(self, *args, **kwargs):
        """Инициализация MBTI препроцессора для постов."""
        super().__init__(*args, **kwargs)
        
        # Паттерн типов MBTI
        self.mbti_pattern = re.compile(r'\b[IE][NS][TF][JP]\b', re.IGNORECASE)
    
    def clean_mbti_post(self, text: str) -> str:
        """
        Очистка MBTI поста от упоминаний типов MBTI.
        
        Параметры:
            text: Входной текст
        
        Возвращает:
            Очищенный текст
        """
        # Удаляем упоминания типов MBTI чтобы избежать утечки данных
        text = self.mbti_pattern.sub('', text)
        return self.clean_text(text)
    
    def preprocess_posts(self, posts: Union[str, List[str]], separator: str = "|||") -> str:
        """
        Предобработка нескольких постов от одного пользователя.
        
        MBTI датасет часто содержит несколько постов на пользователя, разделённых '|||'.
        
        Параметры:
            posts: Одна строка с разделителем или список постов
            separator: Разделитель, используемый в строке постов
        
        Возвращает:
            Предобработанный объединённый текст
        """
        # Разделяем если это одна строка
        if isinstance(posts, str):
            post_list = posts.split(separator)
        else:
            post_list = posts
        
        # Предобрабатываем каждый пост
        all_tokens = []
        for post in post_list:
            post = self.clean_mbti_post(post)
            tokens = self.preprocess(post)
            all_tokens.extend(tokens)
        
        return ' '.join(all_tokens)


def create_preprocessor(
    preset: str = "default",
    **kwargs
) -> TextPreprocessor:
    """
    Фабричная функция для создания препроцессоров с пресетами.
    
    Параметры:
        preset: Пресет конфигурации ('default', 'minimal', 'aggressive', 'mbti')
        **kwargs: Переопределение настроек пресета
    
    Возвращает:
        Экземпляр TextPreprocessor
    """
    presets = {
        "default": {
            "lowercase": True,
            "remove_urls": True,
            "remove_special_chars": True,
            "remove_numbers": False,
            "remove_stopwords": True,
            "lemmatize": True,
            "min_token_length": 2,
        },
        "minimal": {
            "lowercase": True,
            "remove_urls": True,
            "remove_special_chars": False,
            "remove_numbers": False,
            "remove_stopwords": False,
            "lemmatize": False,
            "min_token_length": 1,
        },
        "aggressive": {
            "lowercase": True,
            "remove_urls": True,
            "remove_special_chars": True,
            "remove_numbers": True,
            "remove_stopwords": True,
            "lemmatize": True,
            "min_token_length": 3,
        },
        "mbti": {
            "lowercase": True,
            "remove_urls": True,
            "remove_special_chars": True,
            "remove_numbers": False,
            "remove_stopwords": True,
            "lemmatize": True,
            "min_token_length": 2,
        }
    }
    
    if preset not in presets:
        raise ValueError(f"Неизвестный пресет: {preset}. Доступные: {list(presets.keys())}")
    
    config = presets[preset]
    config.update(kwargs)
    
    if preset == "mbti":
        return MBTIPostPreprocessor(**config)
    else:
        return TextPreprocessor(**config)


if __name__ == "__main__":
    # Пример использования
    print("Тестирование текстового препроцессора...")
    
    # Пример MBTI поста
    sample_text = """
    Hey everyone! Check out this cool article: https://example.com/article
    I'm an INTJ and I really love programming #coding @friend Let's connect!!!
    123 people liked my post... 🚀
    """
    
    # Создаём препроцессор
    preprocessor = create_preprocessor("mbti")
    
    print(f"\nИсходный текст:\n{sample_text}")
    
    # Предобрабатываем
    tokens = preprocessor.preprocess(sample_text)
    print(f"\nТокены: {tokens}")
    
    processed_text = preprocessor.preprocess_to_string(sample_text)
    print(f"\nОбработанный текст: {processed_text}")
    
    # Тест MBTI-специфичной предобработки
    mbti_preprocessor = MBTIPostPreprocessor()
    multi_post = "I love coding!|||Check out my project|||INTJ here"
    processed_posts = mbti_preprocessor.preprocess_posts(multi_post)
    print(f"\nОбработка мульти-постов: {processed_posts}")
    
    print("\n✅ Тест препроцессора пройден!")


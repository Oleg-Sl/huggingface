
# <|system|> [Ваш текст, задающий роль и правила] <|user|> [Вопрос пользователя] <|assistant|> [Ответ модели]



# 1.

def format_qa_pair(system_prompt: str, user_question: str, model_answer: str) -> str:
    """
    Форматирует одну QA-пару в единую строку для LoRA-дообучения.
    """
    # 1. Системная инструкция (константа для всего датасета)
    system_part = "<|system|>" + system_prompt.strip()
    
    # 2. Вопрос пользователя
    user_part = "<|user|>" + user_question.strip()
    
    # 3. Ответ модели (правильный, эталонный ответ)
    assistant_part = "<|assistant|>" + model_answer.strip()
    
    # Объединяем все части, используя пробел как разделитель между токенами
    # (Некоторые фреймворки не требуют пробелов, но это безопаснее)
    template = system_part + " " + user_part + " " + assistant_part
    
    return template

# --- Пример использования ---

SYSTEM_PROMPT = "Ты — эксперт по налоговому законодательству РФ. Отвечай только на основе предоставленных законов и будь максимально краток."

question = "Какие ставки НДФЛ действуют для нерезидентов?"
answer = "Для нерезидентов РФ действует ставка НДФЛ в 30%."

formatted_string = format_qa_pair(SYSTEM_PROMPT, question, answer)

# Вывод будет выглядеть так:
# <|system|>Ты — эксперт по налоговому законодательству РФ...<|user|>Какие ставки НДФЛ...?<|assistant|>Для нерезидентов РФ действует ставка...





# 2.

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# 1.1 Имитация данных (ваша чистая, нормализованная выборка)
data = pd.DataFrame({
    'text_full': [
        "Как работает LoRA, в чем ее суть?", 
        "Какой принцип работы у LoRA? Объясни.", # Дубликат по смыслу
        "Каков возраст нашей планеты Земля?",     # Уникальный пример
    ],
    'id': [101, 102, 103]
})
texts = data['text_full'].tolist()

# 1.2 Выбор специализированной модели для русского языка
# (Замените на выбранную вами модель ru-SBERT)
model = SentenceTransformer('cointegrated/rubert-tiny2') 
embeddings = model.encode(texts, convert_to_tensor=False)
D = embeddings.shape[1] # Размерность вектора (например, 312)


# 2.1 Создание индекса
index = faiss.IndexFlatIP(D) 
index.add(embeddings)

# 2.2 Установка порога, о котором мы говорили
THRESHOLD = 0.95 

duplicates_to_remove = set()

# 2.3 Итерация по каждому вектору
for i in range(len(embeddings)):
    # Ищем 5 ближайших соседей (включая сам элемент)
    D_scores, I_indices = index.search(embeddings[i:i+1], k=5) 
    
    # 2.4 Проверка найденных соседей
    for j, neighbor_index in enumerate(I_indices[0]):
        similarity = D_scores[0][j] # Значение Косинусного сходства

        # Условие: не сам элемент и сходство выше порога
        if i != neighbor_index and similarity >= THRESHOLD:
            
            # Логика: мы удаляем элемент с БОЛЬШИМ ID, чтобы сохранить оригинал
            original_id = data.loc[i, 'id']
            duplicate_id = data.loc[neighbor_index, 'id']
            
            if original_id < duplicate_id:
                duplicates_to_remove.add(duplicate_id) 
            else:
                duplicates_to_remove.add(original_id)



# Финальная очистка DataFrame
cleaned_data = data[~data['id'].isin(duplicates_to_remove)] 

print(f"Оригинал: {len(data)} записей.")
print(f"Удалено дубликатов: {len(duplicates_to_remove)}.")
print(f"Оставлено уникальных записей: {len(cleaned_data)}.")

# Вывод первых нескольких уникальных записей:
# print(cleaned_data.head())



# 4.

import json
# Используем функцию format_qa_pair из предыдущего шага
# (Предполагается, что она определена выше)

def create_training_file(clean_qa_pairs: list, system_prompt: str, output_path: str):
    """Преобразует список чистых QA-пар в файл JSONL для обучения."""
    
    formatted_data = []
    
    # Итерация по всем парам (Вопрос, Ответ)
    for question, answer in clean_qa_pairs:
        
        # 1. Форматируем строку
        formatted_string = format_qa_pair(system_prompt, question, answer)
        
        # 2. Создаем объект JSON
        formatted_data.append({"text": formatted_string})
        
    # 3. Сохраняем в файл JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            # Записываем каждый JSON-объект в отдельную строку
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return output_path

# Пример использования:
# final_file = create_training_file(ваши_чистые_пары, ВАШ_СИСТЕМНЫЙ_ПРОМПТ, "training_data.jsonl")




# 5.

import json
import re
import pandas as pd
# --- Для реального запуска: pip install faiss-cpu sentence-transformers ---
# import faiss
# from sentence_transformers import SentenceTransformer 
# import numpy as np


# ----------------------------------------------
# 1. КОНСТАНТЫ И ФУНКЦИИ ФОРМАТИРОВАНИЯ
# ----------------------------------------------

# Системная инструкция, которую мы обсудили
SYSTEM_PROMPT = "Ты — высококвалифицированный эксперт по налоговому законодательству РФ. Отвечай только на основе предоставленных законов и будь максимально точен."
THRESHOLD = 0.95 # Порог Косинусного сходства для дедупликации

def clean_and_normalize(text: str) -> str:
    """Применяет унификацию пунктуации и очистку."""
    # Приведение к нижнему регистру (для консистентности)
    text = text.lower()
    
    # Замена умных кавычек/тире на стандартные (Нормализация)
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('«', '"').replace('»', '"').replace('“', '"').replace('”', '"')
    
    # Удаление повторяющихся пробелов (Унификация пунктуации)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Добавьте здесь NFKC-нормализацию (unicodedata.normalize('NFKC', text)) для невидимых символов
    
    return text

def format_qa_pair(system_prompt: str, user_question: str, model_answer: str) -> str:
    """Форматирует одну QA-пару в единую строку для обучения."""
    # Используем токены-разделители, принятые в мире LLM
    template = (
        "<|system|>" + system_prompt.strip() + 
        " <|user|>" + user_question.strip() + 
        " <|assistant|>" + model_answer.strip()
    )
    return template


# ----------------------------------------------
# 2. ФУНКЦИЯ СЕМАНТИЧЕСКОЙ ДЕДУПЛИКАЦИИ (КОНЦЕПТ)
# ----------------------------------------------

def perform_semantic_deduplication(qa_pairs: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Концептуальный пример. В реальном коде здесь должна быть логика FAISS
    и SentenceTransformer для поиска и удаления близких по смыслу пар.
    """
    print(f"\n-> Запуск семантической дедупликации с порогом: {threshold}")
    
    # На этом этапе должны быть загружены модель и индекс FAISS,
    # и должен быть выполнен поиск ближайших соседей (ANN).
    
    # Упрощаем: удаляем только один очевидный дубликат для примера
    # В реальном коде: здесь останутся только уникальные по смыслу ID
    
    initial_len = len(qa_pairs)
    # Имитируем удаление одного дубликата
    qa_pairs_cleaned = qa_pairs[qa_pairs['id'] != 102]
    
    print(f"-> Исходный размер: {initial_len}, Размер после дедупликации: {len(qa_pairs_cleaned)}")
    return qa_pairs_cleaned


# ----------------------------------------------
# 3. ГЛАВНЫЙ ПАЙПЛАЙН
# ----------------------------------------------

def main_pipeline(raw_data: list, system_prompt: str, output_file: str):
    """Объединяет все этапы подготовки данных."""
    
    # Шаг 1: Нормализация и подготовка данных
    print("\n[ШАГ 1/4] Нормализация и подготовка...")
    qa_df = pd.DataFrame(raw_data)
    qa_df['question'] = qa_df['question'].apply(clean_and_normalize)
    qa_df['answer'] = qa_df['answer'].apply(clean_and_normalize)
    
    # Для семантической дедупликации объединяем текст
    qa_df['text_full'] = qa_df['question'] + " " + qa_df['answer']
    
    # Шаг 2: Семантическая дедупликация
    qa_df_unique = perform_semantic_deduplication(qa_df, THRESHOLD)
    
    # Шаг 3: Форматирование в финальную строку
    print("\n[ШАГ 3/4] Финальное форматирование...")
    final_data_for_jsonl = []
    
    for index, row in qa_df_unique.iterrows():
        formatted_string = format_qa_pair(
            system_prompt, 
            row['question'], 
            row['answer']
        )
        # Создаем JSON-объект в требуемом формате {"text": "..."}
        final_data_for_jsonl.append({"text": formatted_string})
        
    # Шаг 4: Сохранение в JSONL
    print(f"[ШАГ 4/4] Сохранение в файл: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in final_data_for_jsonl:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            
    print("\n✅ Подготовка данных завершена! Файл готов к загрузке в тренировочный скрипт.")


# --- Исходные данные (Для примера) ---
raw_qa_pairs = [
    {'id': 101, 'question': 'как работает LoRA?', 'answer': 'LoRA добавляет небольшие, обучаемые адаптеры.'},
    {'id': 102, 'question': 'В чем принцип работы LoRA?', 'answer': 'LoRA вводит маленькие, обучаемые адаптеры в модель.'}, # Дубликат по смыслу
    {'id': 103, 'question': 'Какая ставка НДФЛ для резидентов?', 'answer': '13%.'},
    {'id': 104, 'question': 'Какая ставка НДФЛ для резидентов?', 'answer': '13%.'} # Строгий дубликат
]

# Запуск всего процесса
main_pipeline(raw_qa_pairs, SYSTEM_PROMPT, "final_training_data.jsonl")




# 6.
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

def perform_semantic_deduplication(qa_pairs: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Реализует поиск семантических совпадений (дубликатов) с использованием FAISS.
    
    Входные данные (qa_pairs) должны содержать колонки 'text_full' (Вопрос + Ответ) и 'id'.
    """
    print(f"\n[Запуск] Семантическая дедупликация с порогом: {threshold}")

    # --- 1. Векторизация ---
    # Используем концептуальный ru-SBERT (Замените на вашу специализированную модель)
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    
    # Кодируем текст, нормализуем (для корректного Косинусного сходства)
    embeddings = model.encode(qa_pairs['text_full'].tolist(), 
                              convert_to_tensor=False, 
                              normalize_embeddings=True) 
    
    D = embeddings.shape[1] # Размерность вектора
    
    # --- 2. Индексирование FAISS ---
    # IndexFlatIP (Inner Product) эквивалентен Косинусному сходству для нормализованных векторов
    index = faiss.IndexFlatIP(D) 
    index.add(embeddings)

    # --- 3. Поиск и Фильтрация ---
    duplicates_to_remove = set()
    
    # Ищем 5 ближайших соседей (D - сходство, I - индексы)
    # k=5 достаточно, чтобы найти ближайшие дубликаты
    D_scores, I_indices = index.search(embeddings, k=5) 

    for i in range(len(embeddings)):
        neighbor_indices = I_indices[i]
        
        for j, neighbor_idx in enumerate(neighbor_indices):
            similarity = D_scores[i][j] # Значение Косинусного сходства
            
            # Условие: это не сам элемент (i != neighbor_idx) И сходство выше порога
            if i != neighbor_idx and similarity >= threshold:
                
                # Логика удаления: удаляем запись с БОЛЕЕ ВЫСОКИМ ID, чтобы сохранить оригинал
                original_id = qa_pairs.iloc[i]['id']
                duplicate_id = qa_pairs.iloc[neighbor_idx]['id']
                
                # Мы добавляем ID, который нужно удалить
                if original_id < duplicate_id:
                    duplicates_to_remove.add(duplicate_id)
                else:
                    duplicates_to_remove.add(original_id)
                    
    # --- 4. Финальная очистка ---
    initial_len = len(qa_pairs)
    qa_pairs_cleaned = qa_pairs[~qa_pairs['id'].isin(duplicates_to_remove)].copy()
    
    print(f"-> Исходный размер: {initial_len}, Удалено: {len(duplicates_to_remove)}, Оставлено: {len(qa_pairs_cleaned)}")
    return qa_pairs_cleaned





import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

def perform_semantic_deduplication(qa_pairs: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Реализует поиск семантических совпадений (дубликатов) с использованием FAISS.
    
    Входные данные (qa_pairs) должны содержать колонки 'text_full' (Вопрос + Ответ) и 'id'.
    """
    print(f"\n[Запуск] Семантическая дедупликация с порогом: {threshold}")

    # --- 1. Векторизация ---
    # Используем концептуальный ru-SBERT (Замените на вашу специализированную модель)
    model = SentenceTransformer('cointegrated/rubert-tiny2')
    
    # Кодируем текст, нормализуем (для корректного Косинусного сходства)
    embeddings = model.encode(qa_pairs['text_full'].tolist(), 
                              convert_to_tensor=False, 
                              normalize_embeddings=True) 
    
    D = embeddings.shape[1] # Размерность вектора
    
    # --- 2. Индексирование FAISS ---
    # IndexFlatIP (Inner Product) эквивалентен Косинусному сходству для нормализованных векторов
    index = faiss.IndexFlatIP(D) 
    index.add(embeddings)

    # --- 3. Поиск и Фильтрация ---
    duplicates_to_remove = set()
    
    # Ищем 5 ближайших соседей (D - сходство, I - индексы)
    # k=5 достаточно, чтобы найти ближайшие дубликаты
    D_scores, I_indices = index.search(embeddings, k=5) 

    for i in range(len(embeddings)):
        neighbor_indices = I_indices[i]
        
        for j, neighbor_idx in enumerate(neighbor_indices):
            similarity = D_scores[i][j] # Значение Косинусного сходства
            
            # Условие: это не сам элемент (i != neighbor_idx) И сходство выше порога
            if i != neighbor_idx and similarity >= threshold:
                
                # Логика удаления: удаляем запись с БОЛЕЕ ВЫСОКИМ ID, чтобы сохранить оригинал
                original_id = qa_pairs.iloc[i]['id']
                duplicate_id = qa_pairs.iloc[neighbor_idx]['id']
                
                # Мы добавляем ID, который нужно удалить
                if original_id < duplicate_id:
                    duplicates_to_remove.add(duplicate_id)
                else:
                    duplicates_to_remove.add(original_id)
                    
    # --- 4. Финальная очистка ---
    initial_len = len(qa_pairs)
    qa_pairs_cleaned = qa_pairs[~qa_pairs['id'].isin(duplicates_to_remove)].copy()
    
    print(f"-> Исходный размер: {initial_len}, Удалено: {len(duplicates_to_remove)}, Оставлено: {len(qa_pairs_cleaned)}")
    return qa_pairs_cleaned

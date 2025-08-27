#### 1. Подготовка данных (Самый важный этап)

Вам нужен набор текстов (документов), где каждое слово (токен) размечено специальным тегом, указывающим, является ли оно частью ФИО.

*   **Формат разметки BIO (или BIO2):**
    *   **`O`** (Outside) — не является сущностью.
    *   **`B-PER`** (Begin-Person) — начало ФИО.
    *   **`I-PER`** (Inside-Person) — продолжение ФИО (второе или третье слово в ФИО).

**Пример размеченного текста:**

| Текст          | Метка |
| :------------- | :---- |
| Документ       | O     |
| подготовил     | O     |
| **Иванов**     | B-PER |
| **Иван**       | I-PER |
| **Иванович**   | I-PER |
| ,              | O     |
| **менеджер**   | O     |
| **проекта**    | O     |

*   **Как размечать:** Используйте инструменты для разметки, такие как **LabelStudio**, **Brat** или даже простой Excel/Google Tables. Сохраните результат в формате **`.conll`** (текстовый файл, где слова и метки разделены табуляцией, а предложения — пустой строкой) или в **JSON**.

#### 2. Установка необходимых библиотек

```bash
pip install transformers datasets torch accelerate
# Если будете использовать скрипт с Trainer:
pip install transformers[torch]
# Для работы с данными также часто нужны:
pip install pandas numpy scikit-learn
```

#### 3. Код для дообучения (Fine-Tuning)

Создайте Python-скрипт (`train_ner.py`) со следующим содержанием:

```python
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset, load_metric
import numpy as np

# --- CONFIG ---
model_name = "DeepPavlov/xlm-roberta-large-en-ru"  # Ваша модель
output_dir = "./finetuned_ner_model"               # Куда сохранить модель
num_epochs = 3                                     # Количество эпох обучения
batch_size = 8                                     # Размер батча (зависит от памяти GPU)

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 1. Загрузка и подготовка ваших данных
# Предположим, у вас есть функции для загрузки данных в формате CONLL
def load_conll_data(file_path):
    # Эта функция должна парсить ваш .conll файл
    # и возвращать список предложений (каждое предложение - список слов)
    # и список соответствующих меток (каждое предложение - список меток)
    sentences, labels = [], []
    # ... (ваш код для парсинга файла) ...
    return sentences, labels

train_sentences, train_labels = load_conll_data("train.conll")
val_sentences, val_labels = load_conll_data("dev.conll")

# 2. Создание объекта Dataset
# Нам нужно создать словарь с ключами 'id', 'tokens', 'ner_tags'
train_dataset = Dataset.from_dict({
    'tokens': train_sentences,
    'ner_tags': train_labels
})
val_dataset = Dataset.from_dict({
    'tokens': val_sentences,
    'ner_tags': val_labels
})

# 3. Токенизация и выравнивание меток (ВАЖНЕЙШИЙ ШАГ)
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,  # Критически важно для NER!
        padding='max_length',      # Можно использовать 'max_length' или True
        max_length=512
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Сопоставление токенов со словами
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Специальные токены ([CLS], [SEP], [PAD]) получают метку -100
            if word_idx is None:
                label_ids.append(-100)
            # Для первого токена каждого слова назначаем метку
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # Для подтокенов одного слова назначаем -100
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Применяем функцию ко всему датасету
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

# 4. Загрузка модели
# Укажите id2label и label2id в соответствии с вашими метками
id2label = {0: "O", 1: "B-PER", 2: "I-PER"}
label2id = {"O": 0, "B-PER": 1, "I-PER": 2}

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True  # Полезно, если число меток не совпадает со стандартным
)

# 5. Настройка тренировки
args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# 6. Запуск обучения
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# 7. Сохранение модели
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Модель сохранена в {output_dir}")
```

#### 4. Запуск обучения

Выполните в командной строке:
```bash
python train_ner.py
```

#### 5. Использование дообученной модели

После обучения используйте модель для предсказания:

```python
from transformers import pipeline

# Укажите путь к вашей дообученной модели
ner_pipeline = pipeline(
    "token-classification",
    model="./finetuned_ner_model",
    aggregation_strategy="simple"  # Группирует подтокены в слова
)

text = "Документ подготовил Иванов Иван Иванович, менеджер проекта."
results = ner_pipeline(text)

for entity in results:
    print(f"Сущность: {entity['word']}, Метка: {entity['entity_group']}, Score: {entity['score']:.4f}")
```


## 1. Установка необходимых библиотек

```bash
pip install transformers datasets torch sklearn seqeval accelerate
```

## 2. Подготовка данных для обучения

```python
from datasets import Dataset, ClassLabel
import pandas as pd
from transformers import AutoTokenizer

# Пример данных: текст и соответствующие темы
data = {
    "text": [
        "Российская экономика показывает рост в четвертом квартале. ВВП увеличился на 3.5%.",
        "Новые технологии искусственного интеллекта revolutionize медицинскую диагностику.",
        "Футбольный клуб Спартак одержал победу в московском дерби со счетом 2:1.",
        "Климатические изменения приводят к экстремальным погодным условиям по всему миру.",
        "Банковский сектор внедряет новые цифровые решения для улучшения обслуживания клиентов."
    ],
    "topic": [
        "экономика",
        "технологии", 
        "спорт",
        "наука",
        "финансы"
    ]
}

# Создаем датасет
dataset = Dataset.from_pandas(pd.DataFrame(data))

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-base-cased")

# Определяем метки классов
class_labels = ClassLabel(names=list(set(data["topic"])))

def tokenize_function(examples):
    # Токенизация текста
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Преобразование тем в числовые метки
    labels = [class_labels.str2int(topic) for topic in examples["topic"]]
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }

# Применяем токенизацию
tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

## 3. Разделение данных на train/validation

```python
# Разделяем данные
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
```

## 4. Создание модели и настройка обучения

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Загружаем модель для классификации
model = AutoModelForSequenceClassification.from_pretrained(
    "cointegrated/rubert-base-cased",
    num_labels=len(class_labels.names),
    id2label={i: label for i, label in enumerate(class_labels.names)},
    label2id={label: i for i, label in enumerate(class_labels.names)}
)

# Функция для вычисления метрик
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    
    # Дополнительные метрики
    report = classification_report(labels, predictions, 
                                 target_names=class_labels.names,
                                 output_dict=True)
    
    return {
        "accuracy": accuracy,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }

# Аргументы обучения
training_args = TrainingArguments(
    output_dir="./rubert-topic-classification",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=10,
)
```

## 5. Запуск обучения

```python
# Создаем тренер
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Запускаем обучение
trainer.train()

# Сохраняем модель
trainer.save_model("./rubert-topic-classification-final")
```

## 6. Пример использования обученной модели

```python
from transformers import pipeline

# Создаем pipeline для классификации
classifier = pipeline(
    "text-classification",
    model="./rubert-topic-classification-final",
    tokenizer=tokenizer
)

# Тестируем на новых текстах
test_texts = [
    "Акции компании показали рекордный рост на бирже",
    "Ученые разработали новую вакцину против вируса",
    "Баскетбольная команда вышла в финал чемпионата"
]

for text in test_texts:
    result = classifier(text)
    print(f"Текст: {text}")
    print(f"Предсказанная тема: {result[0]['label']} (вероятность: {result[0]['score']:.3f})")
    print("-" * 50)
```

## 7. Альтернативный способ - ручное предсказание

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def predict_topic(text, model_path="./rubert-topic-classification-final"):
    # Загружаем модель и токенизатор
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Токенизируем текст
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Предсказание
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = predictions.argmax().item()
        confidence = predictions.max().item()
    
    # Получаем название темы
    predicted_topic = model.config.id2label[predicted_class_id]
    
    return predicted_topic, confidence

# Пример использования
text = "Центральный банк объявил о новых мерах денежно-кредитной политики"
topic, confidence = predict_topic(text)
print(f"Текст: {text}")
print(f"Главная тема: {topic} (уверенность: {confidence:.3f})")
```

## 8. Дополнительная настройка для больших данных

Если у вас большой объем данных, рекомендуется использовать:

```python
# Для ускорения обучения
training_args = TrainingArguments(
    # ... предыдущие параметры ...
    fp16=True,  # Использовать mixed precision
    gradient_accumulation_steps=2,
    dataloader_num_workers=4,
)
```

Этот пример показывает полный pipeline дообучения модели rubert-base-cased для задачи классификации тем документов. Вы можете адаптировать его под свои данные и требования.

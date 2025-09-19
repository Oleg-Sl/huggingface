Для распознавания таблиц из PDF в Excel есть несколько эффективных подходов. Вот лучшие решения:

## 1. **Tesseract + OpenCV для обнаружения таблиц**

```python
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
import os

def detect_tables(image):
    """Обнаружение таблиц на изображении"""
    # Конвертируем в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Бинаризация
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Вертикальные линии
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Горизонтальные линии
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Комбинируем линии
    table_mask = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    
    return table_mask

def extract_table_from_region(image, region):
    """Извлечение таблицы из региона"""
    x, y, w, h = region
    roi = image[y:y+h, x:x+w]
    
    # Используем Tesseract для распознавания с указанием структуры
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    data = pytesseract.image_to_data(roi, config=custom_config, output_type=pytesseract.Output.DICT)
    
    # Собираем текст в структурированный формат
    cells = {}
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:  # Фильтр по уверенности
            text = data['text'][i].strip()
            if text:
                block_num = data['block_num'][i]
                line_num = data['line_num'][i]
                word_num = data['word_num'][i]
                
                if block_num not in cells:
                    cells[block_num] = {}
                if line_num not in cells[block_num]:
                    cells[block_num][line_num] = []
                
                cells[block_num][line_num].append(text)
    
    # Преобразуем в таблицу
    table_data = []
    for block in sorted(cells.keys()):
        for line in sorted(cells[block].keys()):
            table_data.append(cells[block][line])
    
    return table_data

def pdf_tables_to_excel(pdf_path, output_excel_path):
    """Основная функция для конвертации PDF с таблицами в Excel"""
    # Конвертируем PDF в изображения
    images = convert_from_path(pdf_path, dpi=300)
    
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        for page_num, image in enumerate(images):
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Обнаруживаем таблицы
            table_mask = detect_tables(img_cv)
            
            # Находим контуры таблиц
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 1000:  # Игнорируем маленькие области
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Извлекаем таблицу
                    table_data = extract_table_from_region(img_cv, (x, y, w, h))
                    
                    if table_data:
                        # Создаем DataFrame
                        max_cols = max(len(row) for row in table_data)
                        for row in table_data:
                            while len(row) < max_cols:
                                row.append('')
                        
                        df = pd.DataFrame(table_data)
                        
                        # Сохраняем в Excel
                        sheet_name = f'Page_{page_num+1}_Table_{i+1}'
                        df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    
    print(f"Таблицы сохранены в: {output_excel_path}")

# Использование
pdf_tables_to_excel('document.pdf', 'tables.xlsx')
```

## 2. **Использование Camelot (специализированная библиотека)**

```python
import camelot
import pandas as pd

def camelot_pdf_to_excel(pdf_path, output_excel_path):
    """Использование Camelot для извлечения таблиц"""
    # Извлекаем таблицы
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
    
    print(f"Найдено {tables.n} таблиц")
    
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        for i, table in enumerate(tables):
            # Конвертируем в DataFrame
            df = table.df
            
            # Очищаем данные
            df = df.replace('\n', ' ', regex=True)
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            
            # Сохраняем
            sheet_name = f'Table_{i+1}'
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
            print(f"Таблица {i+1}: {df.shape[0]} строк, {df.shape[1]} столбцов")
    
    return tables.n

# Установка: pip install camelot-py[cv] openpyxl
```

## 3. **Tabula-py (альтернатива Camelot)**

```python
import tabula
import pandas as pd

def tabula_pdf_to_excel(pdf_path, output_excel_path):
    """Использование Tabula для извлечения таблиц"""
    # Извлекаем все таблицы
    tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
    
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        for i, df in enumerate(tables):
            if not df.empty:
                # Очистка данных
                df = df.dropna(how='all').dropna(axis=1, how='all')
                df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                
                sheet_name = f'Table_{i+1}'
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return len(tables)

# Установка: pip install tabula-py openpyxl
```

## 4. **Комбинированный подход с предобработкой**

```python
def enhanced_table_extraction(pdf_path, output_excel_path):
    """Улучшенное извлечение таблиц с предобработкой"""
    images = convert_from_path(pdf_path, dpi=400)
    
    all_tables = []
    
    for page_num, image in enumerate(images):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Улучшаем качество изображения
        enhanced = enhance_image_quality(img_cv)
        
        # Пробуем несколько методов
        try:
            # Метод 1: Camelot
            tables_camelot = camelot.read_pdf(pdf_path, pages=str(page_num+1))
            for table in tables_camelot:
                all_tables.append(table.df)
        except:
            pass
        
        try:
            # Метод 2: Tesseract с обнаружением структуры
            custom_config = r'--oem 3 --psm 6 -c tessedit_write_structured_text=1'
            osd = pytesseract.image_to_osd(enhanced)
            # Анализ ориентации и структуры
        except:
            pass
    
    # Сохраняем все таблицы в Excel
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        for i, table_df in enumerate(all_tables):
            sheet_name = f'Table_{i+1}'
            table_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return len(all_tables)

def enhance_image_quality(image):
    """Улучшение качества изображения для лучшего распознавания"""
    # Увеличение контраста
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return enhanced
```

## Установка необходимых библиотек:

```bash
# Основные зависимости
pip install camelot-py[cv] tabula-py pytesseract pdf2image opencv-python pandas openpyxl

# Дополнительно
pip install pillow numpy

# Установка Tesseract (Windows)
# Скачать: https://github.com/UB-Mannheim/tesseract/wiki
# Установить русский язык: https://github.com/tesseract-ocr/tessdata
```

## Рекомендации:

1. **Camelot** - лучший выбор для четко очерченных таблиц
2. **Tabula** - хорош для простых таблиц
3. **Tesseract + OpenCV** - для сложных случаев и кастомной обработки
4. Используйте высокое DPI (300-400) для качественного распознавания
5. Для русских текстов убедитесь в наличии языковых пакетов Tesseract

Начните с Camelot, если таблицы имеют четкие границы, или с комбинированного подхода для сложных случаев.

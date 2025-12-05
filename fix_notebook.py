import json
import sys

def fix_notebook(input_file, output_file):
    """Исправляет ноутбук для GitHub"""
    with open(input_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Удаляем проблемные метаданные
    if 'metadata' in notebook:
        notebook['metadata'].pop('widgets', None)
        notebook['metadata'].pop('colab', None)
    
    # Очищаем метаданные ячеек
    for cell in notebook['cells']:
        if 'metadata' in cell:
            cell['metadata'] = {}
    
    # Сохраняем исправленный файл
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"Ноутбук исправлен: {output_file}")

if __name__ == "__main__":
    fix_notebook('car_price_prediction.ipynb', 'car_price_prediction_fixed.ipynb')
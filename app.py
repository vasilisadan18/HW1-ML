import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Функции для предобработки данных
def clean_numeric_string(value):
    """Очистка строковых числовых значений"""
    if pd.isna(value):
        return np.nan
    
    if isinstance(value, (int, float)):
        return float(value)
    
    value_str = str(value).strip()
    
    cleaned = ''.join(char for char in value_str
                     if char.isdigit() or char == '.' or char == '-')
    
    if not cleaned or cleaned == '-' or cleaned == '.':
        return np.nan
    
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

def preprocess_input_data(input_df, artifacts):
    """Предобработка введенных данных"""
    df = input_df.copy()
    
    # Очищаем числовые столбцы
    columns_to_clean = ['mileage', 'engine', 'max_power']
    for column in columns_to_clean:
        if column in df.columns:
            df[column] = df[column].apply(clean_numeric_string)
    
    # Заполняем пропуски медианами из артефактов
    if 'medians' in artifacts:
        medians = artifacts['medians']
        for column in columns_to_clean + ['seats']:
            if column in df.columns and column in medians:
                df[column] = df[column].fillna(medians[column])
    
    return df

def prepare_features_for_model(model_name, input_df, artifacts):
    """Подготовка признаков для конкретной модели"""
    df = input_df.copy()
    
    # Для всех моделей кроме ridge - только числовые признаки
    if model_name in ['linear_regression', 'lasso', 'elastic_net']:
        # Только числовые признаки
        numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
        features = df[numeric_features].copy()
        
        # Масштабируем для lasso и elastic_net
        if model_name in ['lasso', 'elastic_net'] and 'scaler' in artifacts:
            scaler = artifacts['scaler']
            features_scaled = scaler.transform(features)
            features = pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
        
        return features
    
    # Для ridge - создаем признаки как в обучении
    elif model_name == 'ridge':
        # Получаем модель Ridge
        if 'models' in artifacts and 'ridge' in artifacts['models']:
            ridge_model = artifacts['models']['ridge']
            
            # Пытаемся получить имена признаков из модели
            if hasattr(ridge_model, 'feature_names_in_'):
                # Если модель сохранила имена признаков
                expected_features = list(ridge_model.feature_names_in_)
                
                # Создаем DataFrame с нулями
                features = pd.DataFrame(0, index=df.index, columns=expected_features)
                
                # Заполняем числовые признаки
                numeric_mapping = {
                    'year': 'year',
                    'km_driven': 'km_driven',
                    'mileage': 'mileage',
                    'engine': 'engine',
                    'max_power': 'max_power',
                    'seats': 'seats'
                }
                
                for model_feat, input_feat in numeric_mapping.items():
                    if model_feat in expected_features and input_feat in df.columns:
                        features[model_feat] = df[input_feat]
                
                
                # Заполняем OneHot признаки на основе введенных данных
                if 'fuel' in df.columns:
                    fuel_value = df['fuel'].iloc[0]
                    if f'fuel_{fuel_value}' in expected_features:
                        features[f'fuel_{fuel_value}'] = 1
                
                if 'seller_type' in df.columns:
                    seller_value = df['seller_type'].iloc[0]
                    if f'seller_type_{seller_value}' in expected_features:
                        features[f'seller_type_{seller_value}'] = 1
                
                if 'transmission' in df.columns:
                    trans_value = df['transmission'].iloc[0]
                    if f'transmission_{trans_value}' in expected_features:
                        features[f'transmission_{trans_value}'] = 1
                
                if 'owner' in df.columns:
                    owner_value = df['owner'].iloc[0]
                    if f'owner_{owner_value}' in expected_features:
                        features[f'owner_{owner_value}'] = 1
                
                # Особенная обработка для seats
                if 'seats' in df.columns:
                    seats_value = df['seats'].iloc[0]
                    seats_str = f'seats_{seats_value}.0'
                    if seats_str in expected_features:
                        features[seats_str] = 1
                
                return features
            else:

                # Стандартные 24 признака для Ridge
                expected_features = [
                    'year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
                    'fuel_Petrol', 'fuel_Diesel', 'fuel_CNG', 'fuel_LPG',
                    'seller_type_Individual', 'seller_type_Dealer', 'seller_type_Trustmark Dealer',
                    'transmission_Manual', 'transmission_Automatic',
                    'owner_First Owner', 'owner_Second Owner', 'owner_Third Owner',
                    'owner_Fourth & Above Owner', 'owner_Test Drive Car',
                    'seats_2.0', 'seats_4.0', 'seats_5.0', 'seats_6.0',
                    'seats_7.0', 'seats_8.0', 'seats_9.0', 'seats_10.0', 'seats_14.0'
                ]
                
                # Оставляем только первые 24
                expected_features = expected_features[:24]
                
                features = pd.DataFrame(0, index=df.index, columns=expected_features)
                
                # Заполняем числовые признаки
                numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
                for feat in numeric_features:
                    if feat in features.columns and feat in df.columns:
                        features[feat] = df[feat]
                
                # Заполняем OneHot признаки
                for col in ['fuel', 'seller_type', 'transmission', 'owner']:
                    if col in df.columns:
                        value = df[col].iloc[0]
                        onehot_col = f'{col}_{value}'
                        if onehot_col in features.columns:
                            features[onehot_col] = 1
                
                # Особенная обработка для seats (OneHot)
                if 'seats' in df.columns:
                    seats_value = df['seats'].iloc[0]
                    seats_onehot = f'seats_{seats_value}.0'
                    if seats_onehot in features.columns:
                        features[seats_onehot] = 1
                
                return features
        
        return None
    
    return None

# Загрузка данных и моделей
@st.cache_resource
def load_data_and_models():
    try:
        # Загрузка тренировочных данных
        df_train_raw = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
        
        # Базовая предобработка для EDA
        df_train = df_train_raw.copy()
        
        # Удаляем ненужные колонки
        if 'name' in df_train.columns:
            df_train = df_train.drop('name', axis=1)
        if 'torque' in df_train.columns:
            df_train = df_train.drop('torque', axis=1, errors='ignore')
        
        # Очищаем числовые столбцы
        columns_to_clean = ['mileage', 'engine', 'max_power']
        for column in columns_to_clean:
            if column in df_train.columns:
                df_train[column] = df_train[column].apply(clean_numeric_string)
        
        # Сохраняем медианы
        medians = {}
        for column in columns_to_clean + ['seats']:
            if column in df_train.columns:
                medians[column] = df_train[column].median()
                df_train[column] = df_train[column].fillna(medians[column])
        
        # Загружаем модели из pickle файла
        with open('car_price_models.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        # Добавляем медианы 
        artifacts['medians'] = medians
        
        # Создаем скалер для lasso/elastic_net
        if 'scaler' not in artifacts:
            numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
            df_numeric = df_train[numeric_features].copy()
            scaler = StandardScaler()
            scaler.fit(df_numeric)
            artifacts['scaler'] = scaler
        
        # Сохраняем обработанные данные для EDA
        artifacts['df_train_clean'] = df_train
        
        # информация о Ridge модели
        if 'models' in artifacts and 'ridge' in artifacts['models']:
            ridge_model = artifacts['models']['ridge']
            if hasattr(ridge_model, 'feature_names_in_'):
                artifacts['ridge_feature_names'] = list(ridge_model.feature_names_in_)
        
        return artifacts
    
    except FileNotFoundError as e:
        st.error(f"Файл не найден: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Ошибка при загрузке: {str(e)}")
        return None

# Основной заголовок
st.title("Car Price Prediction App")
st.markdown("---")

# Загрузка данных
artifacts = load_data_and_models()

if artifacts is not None:
    # Сайдбар для навигации
    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Выберите раздел:",
        ["EDA и визуализации", "Предсказание цены", "Коэффициенты моделей"]
    )
    
    # СТРАНИЦА 1: EDA и визуализации 
    if page == "EDA и визуализации":
        st.header("Exploratory Data Analysis (EDA)")
        
        if 'df_train_clean' in artifacts:
            df_train = artifacts['df_train_clean']
            
            # Основная информация
            st.subheader("Основная информация о данных")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Количество строк", f"{len(df_train):,}")
            with col2:
                st.metric("Количество признаков", len(df_train.columns))
            with col3:
                if 'selling_price' in df_train.columns:
                    avg_price = df_train['selling_price'].mean()
                    st.metric("Средняя цена", f"{avg_price:,.0f} руб.")
            
            # Выбор типа визуализации
            viz_type = st.selectbox(
                "Выберите тип визуализации:",
                ["Обзор данных", "Распределения признаков", "Корреляционный анализ"]
            )
            
            # 1. Обзор данных
            if viz_type == "Обзор данных":
                st.subheader(" Обзор данных")
                
                # Показываем первые строки
                rows_to_show = st.slider("Количество строк для отображения:", 5, 50, 10)
                st.write(f"**Первые {rows_to_show} строк:**")
                st.dataframe(df_train.head(rows_to_show))
                
                # Основные статистики
                st.write("**Основные статистики числовых признаков:**")
                numeric_cols = df_train.select_dtypes(include=[np.number]).columns
                st.dataframe(df_train[numeric_cols].describe())
                
                # Информация о категориальных признаках
                st.write("**Категориальные признаки:**")
                categorical_cols = df_train.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    unique_count = df_train[col].nunique()
                    st.write(f"- **{col}**: {unique_count} уникальных значений")
            
            # 2. Распределения признаков
            elif viz_type == "Распределения признаков":
                st.subheader("Распределения признаков")
                
                # Выбор признака
                numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
                if 'selling_price' in numeric_cols:
                    numeric_cols.remove('selling_price')
                
                selected_col = st.selectbox("Выберите признак для анализа:", numeric_cols)
                
                # Создаем график
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Гистограмма
                ax1.hist(df_train[selected_col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='skyblue')
                ax1.set_title(f'Распределение {selected_col}')
                ax1.set_xlabel(selected_col)
                ax1.set_ylabel('Частота')
                ax1.grid(alpha=0.3)
                
                # Scatter plot с ценой
                if 'selling_price' in df_train.columns:
                    ax2.scatter(df_train[selected_col], df_train['selling_price'], alpha=0.5, s=10)
                    ax2.set_title(f'Зависимость цены от {selected_col}')
                    ax2.set_xlabel(selected_col)
                    ax2.set_ylabel('Цена продажи')
                    ax2.grid(alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'Целевая переменная не найдена', 
                            ha='center', va='center', transform=ax2.transAxes)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Статистики по выбранному признаку
                st.write(f"**Статистики для {selected_col}:**")
                col_stats = df_train[selected_col].describe()
                st.write(col_stats)
            
            # 3. Корреляционный анализ
            elif viz_type == "Корреляционный анализ":
                st.subheader("Корреляционный анализ")
                
                numeric_cols = df_train.select_dtypes(include=[np.number]).columns
                
                if len(numeric_cols) > 1:
                    # Корреляционная матрица
                    correlations = df_train[numeric_cols].corr()
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    mask = np.triu(np.ones_like(correlations, dtype=bool))
                    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0,
                               square=True, fmt='.2f', mask=mask,
                               ax=ax, cbar_kws={'shrink': 0.8})
                    ax.set_title('Корреляционная матрица')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Наиболее коррелированные признаки
                    if 'selling_price' in correlations.columns:
                        st.write("**Корреляция признаков с ценой:**")
                        price_corr = correlations['selling_price'].drop('selling_price').sort_values(ascending=False)
                        
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        bars = ax2.barh(price_corr.index, price_corr.values)
                        
                        # Цвета в зависимости от знака
                        for i, bar in enumerate(bars):
                            if price_corr.values[i] >= 0:
                                bar.set_color('green')
                            else:
                                bar.set_color('red')
                        
                        ax2.set_xlabel('Корреляция с ценой')
                        ax2.set_title('Влияние признаков на цену')
                        ax2.grid(axis='x', alpha=0.3)
                        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                        
                        plt.tight_layout()
                        st.pyplot(fig2)
                        
                        # Таблица с корреляциями
                        st.write("**Таблица корреляций:**")
                        corr_df = pd.DataFrame({
                            'Признак': price_corr.index,
                            'Корреляция с ценой': price_corr.values
                        })
                        st.dataframe(corr_df)
    
    # СТРАНИЦА 2: Предсказание цены 
    elif page == "Предсказание цены":
        st.header("Предсказание цены автомобиля")
        
        # Выбор способа ввода данных
        input_method = st.radio(
            "Выберите способ ввода данных:",
            ["Ручной ввод", "Загрузка CSV файла"]
        )
        
        if input_method == "Ручной ввод":
            st.subheader("Введите параметры автомобиля:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                year = st.number_input("Год выпуска", min_value=1980, max_value=2025, value=2015)
                km_driven = st.number_input("Пробег (км)", min_value=0, value=50000)
                mileage = st.number_input("Расход топлива (kmpl)", min_value=0.0, value=20.0)
                engine = st.number_input("Объем двигателя (CC)", min_value=500, value=1500)
            
            with col2:
                max_power = st.number_input("Мощность (bhp)", min_value=0.0, value=100.0)
                seats = st.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9, 10, 14])
                fuel = st.selectbox("Тип топлива", ["Petrol", "Diesel", "CNG", "LPG"])
                seller_type = st.selectbox("Тип продавца", ["Individual", "Dealer", "Trustmark Dealer"])
                transmission = st.selectbox("Трансмиссия", ["Manual", "Automatic"])
                owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", 
                                                 "Fourth & Above Owner", "Test Drive Car"])
            
            # Создание DataFrame из введенных данных
            input_data = pd.DataFrame({
                'year': [year],
                'km_driven': [km_driven],
                'fuel': [fuel],
                'seller_type': [seller_type],
                'transmission': [transmission],
                'owner': [owner],
                'mileage': [mileage],
                'engine': [engine],
                'max_power': [max_power],
                'seats': [seats]
            })
            
            if st.button("Предсказать цену", type="primary"):
                try:
                    # Предобработка данных
                    input_processed = preprocess_input_data(input_data, artifacts)
                    
                    # Предсказания всеми моделями
                    predictions = {}
                    
                    # Для каждой модели
                    for model_name in ['linear_regression', 'lasso', 'elastic_net', 'ridge']:
                        if model_name in artifacts.get('models', {}):
                            model = artifacts['models'][model_name]
                            
                            # Подготавливаем признаки для конкретной модели
                            features = prepare_features_for_model(model_name, input_processed, artifacts)
                            
                            if features is not None:
                                try:
                                    pred = model.predict(features)[0]
                                    predictions[model_name] = pred
                                except Exception as e:
                                    st.warning(f"Ошибка предсказания для {model_name}: {str(e)}")
                    
                    # Отображение результатов
                    st.subheader("Результаты предсказаний:")
                    
                    if predictions:
                        # Создаем DataFrame для отображения
                        results_df = pd.DataFrame({
                            'Модель': list(predictions.keys()),
                            'Предсказанная цена': list(predictions.values())
                        })
                        
                        # Сортируем по цене
                        results_df = results_df.sort_values('Предсказанная цена')
                        
                        # Показываем результаты
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Предсказания моделей:**")
                            
                            # Красивое отображение цен
                            for _, row in results_df.iterrows():
                                price = row['Предсказанная цена']
                                if price > 0 and price < 1e9:  # Реалистичные цены
                                    st.success(f"**{row['Модель']}**: {price:,.0f} руб.")
                                else:
                                    st.error(f"**{row['Модель']}**: {price:,.0f} руб. (некорректное значение)")
                        
                        with col2:
                            # Визуализация только реалистичных значений
                            realistic_preds = results_df[
                                (results_df['Предсказанная цена'] > 0) & 
                                (results_df['Предсказанная цена'] < 1e9)
                            ]
                            
                            if len(realistic_preds) > 0:
                                fig, ax = plt.subplots(figsize=(8, 6))
                                
                                models = realistic_preds['Модель']
                                prices = realistic_preds['Предсказанная цена']
                                
                                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(models)]
                                bars = ax.bar(models, prices, color=colors)
                                
                                ax.set_ylabel('Цена (руб)')
                                ax.set_title('Предсказания моделей')
                                ax.grid(axis='y', alpha=0.3)
                                
                                # Форматируем подписи цен
                                for bar in bars:
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2., height,
                                           f'{height:,.0f}', ha='center', va='bottom', fontsize=10)
                                
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Средняя цена
                                avg_price = realistic_preds['Предсказанная цена'].mean()
                                st.metric("Средняя предсказанная цена", f"{avg_price:,.0f} руб.")
                            else:
                                st.warning("Все модели дали некорректные предсказания")
                    
                    else:
                        st.error("Не удалось получить предсказания")
                
                except Exception as e:
                    st.error(f"Ошибка при предсказании: {str(e)}")
        
        else:  # CSV загрузка
            st.subheader("Загрузите CSV файл с данными")
            
            uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    # Чтение CSV
                    df_csv = pd.read_csv(uploaded_file)
                    st.success(f"Успешно загружено {len(df_csv)} строк")
                    
                    # Показываем предпросмотр
                    st.write("**Предпросмотр данных:**")
                    st.dataframe(df_csv.head())
                    
                    # Проверка необходимых колонок
                    required_cols = ['year', 'km_driven', 'fuel', 'seller_type', 
                                    'transmission', 'owner', 'mileage', 'engine', 
                                    'max_power', 'seats']
                    
                    missing_cols = [col for col in required_cols if col not in df_csv.columns]
                    
                    if missing_cols:
                        st.error(f"Отсутствуют колонки: {missing_cols}")
                    else:
                        if st.button(" Предсказать цены", type="primary"):
                            with st.spinner("Обработка данных..."):
                                # Предобработка всех строк
                                df_processed = preprocess_input_data(df_csv, artifacts)
                                
                                # Собираем все предсказания
                                all_predictions = {}
                                
                                for model_name in ['linear_regression', 'lasso', 'elastic_net', 'ridge']:
                                    if model_name in artifacts.get('models', {}):
                                        model = artifacts['models'][model_name]
                                        features = prepare_features_for_model(model_name, df_processed, artifacts)
                                        
                                        if features is not None:
                                            try:
                                                preds = model.predict(features)
                                                all_predictions[model_name] = preds
                                            except Exception as e:
                                                st.warning(f"Ошибка для {model_name}: {str(e)}")
                                
                                if all_predictions:
                                    # Создаем DataFrame
                                    results_df = df_csv.copy()
                                    
                                    for model_name, preds in all_predictions.items():
                                        results_df[f'pred_{model_name}'] = preds
                                    
                                    # Добавляем среднее предсказание
                                    pred_cols = [f'pred_{m}' for m in all_predictions.keys()]
                                    results_df['pred_average'] = results_df[pred_cols].mean(axis=1)
                                    
                                    st.subheader("Результаты предсказаний:")
                                    
                                    # Показываем первые 5 строк
                                    st.dataframe(results_df.head())
                                    
                
                except Exception as e:
                    st.error(f"Ошибка при обработке файла: {str(e)}")
    
    # СТРАНИЦА 3: Коэффициенты моделей 
    elif page == "Коэффициенты моделей":
        st.header("Коэффициенты моделей")
        
        # Выбор модели
        model_names = list(artifacts.get('models', {}).keys())
        
        if model_names:
            selected_model = st.selectbox("Выберите модель:", model_names)
            
            model = artifacts['models'][selected_model]
            
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
                intercept = getattr(model, 'intercept_', 0)
                
                # Определяем имена признаков
                if selected_model == 'ridge' and hasattr(model, 'feature_names_in_'):
                    feature_names = list(model.feature_names_in_)
                elif selected_model == 'ridge' and 'ridge_feature_names' in artifacts:
                    feature_names = artifacts['ridge_feature_names']
                else:
                    feature_names = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']
                
                # Создаем DataFrame
                coef_df = pd.DataFrame({
                    'Признак': feature_names[:len(coefficients)],
                    'Коэффициент': coefficients,
                    '|Коэффициент|': np.abs(coefficients)
                }).sort_values('|Коэффициент|', ascending=False)
                
                st.subheader(f"Коэффициенты модели {selected_model}")
                
                # Показываем коэффициенты
                st.dataframe(coef_df.style.format({'Коэффициент': '{:.4f}', '|Коэффициент|': '{:.4f}'}))
                
                # Визуализация
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Берем топ-20 признаков
                top_n = min(20, len(coef_df))
                top_coefs = coef_df.head(top_n)
                
                # Цвета
                colors = ['green' if x > 0 else 'red' for x in top_coefs['Коэффициент']]
                
                bars = ax.barh(top_coefs['Признак'], top_coefs['Коэффициент'], color=colors)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xlabel('Значение коэффициента')
                ax.set_title(f'Топ-{top_n} самых важных признаков')
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("У модели нет коэффициентов")
        else:
            st.warning("Модели не найдены")
    

# Футер
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Car Price Prediction App</p>
    </div>
    """,
    unsafe_allow_html=True
)
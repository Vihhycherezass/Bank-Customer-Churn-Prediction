# Прогнозирование оттока клиентов банка

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-3.0+-green.svg)
![Status](https://img.shields.io/badge/status-completed-brightgreen.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.877-success.svg)

Pet-проект по бинарной классификации оттока клиентов банка на основе демографических, финансовых и поведенческих признаков. Выполнен углублённый Feature Engineering, обучена модель LightGBM с подбором гиперпараметров. Достигнут ROC-AUC **0.877**, Recall **0.782**.

---

## Содержание
- [Технологии](#технологии)
- [Начало работы](#начало-работы)
- [Использование](#использование)
- [Разработка](#разработка)
- [Результаты](#результаты)
- [To do](#to-do)
- [Команда проекта](#команда-проекта)
- [Источники](#источники)

---

### Технологии
- Python 3.8+
- pandas, NumPy
- scikit-learn (StandardScaler, OneHotEncoder, RandomizedSearchCV)
- LightGBM
- Matplotlib, Seaborn
- imbalanced-learn (опционально)

---

### Начало работы

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn imbalanced-learn
```
--- 

## Использование

Запустите Jupyter Notebook или Python-скрипт. Основные шаги:

- Загрузка датасета `Customer-Churn-Records.csv`.
- Удаление бесполезных колонок (`RowNumber`, `CustomerId`, `Surname`).
- Создание новых признаков (финансовые метрики, поведенческие индексы, сегментация).
- Кодирование категориальных переменных (`LabelEncoder`, `OneHotEncoder`).
- Стандартизация числовых признаков.
- Обучение модели LightGBM с подбором гиперпараметров через `RandomizedSearchCV`.

---

## Разработка

Пример настройки и обучения модели:

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV

lgbm_base = LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1)

param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 7, 10],
    'num_leaves': [20, 31, 50],
    'subsample': [0.7, 0.8, 0.9]
}

random_search = RandomizedSearchCV(
    lgbm_base, param_dist, n_iter=50, cv=5,
    scoring='roc_auc', random_state=42, n_jobs=-1
)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
```

---

## Результаты

- **ROC-AUC**: 0.877
- **Accuracy**: 0.816
- **Recall**: 0.782
- **Precision**: 0.533
- **F1-Score**: 0.634

Модель хорошо выявляет уходящих клиентов (высокий Recall), что важно для бизнеса. Precision ниже из‑за дисбаланса классов (20% оттока).

---

## To do

- [x] Первичный EDA и очистка данных
- [x] Feature Engineering (13 новых признаков)
- [x] Удаление признаков с утечкой данных
- [x] Подбор гиперпараметров LightGBM
- [ ] Применение SMOTE для балансировки
- [ ] Калибровка вероятностей (CalibratedClassifierCV)
- [ ] Тестирование ансамблей (Voting / Stacking)

---

## Команда проекта

Соло-разработчик: [Назар](https://github.com/Vihhycherezass)

---

## Источники

- Датасет: `Customer-Churn-Records.csv`
- Вдохновение: реальные задачи банковского скоринга и удержания клиентов

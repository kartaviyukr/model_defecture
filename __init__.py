"""
model_defecture — система прогнозирования дефектуры (OOS) лекарственных препаратов.

Прогнозирует вероятность того, что препарат окажется в дефектуре
(остаток < 10% исторической медианы) в течение следующих 14 дней.

Модель: CatBoostClassifier + Optuna (PR-AUC 0.72, ROC-AUC 0.92)
Признаки: 155+ (скорость продаж, дни запаса, риск-флаги, временные паттерны)
Дистрибьюторы: Пульс, Катрен, Протек, ФармаСклад, ГК

Структура модулей
-----------------
helpers.py              — загрузка конфига (YAML + .env), утилиты
logger.py               — настройка loguru (консоль + файл с ротацией)
db_connector.py         — подключение к MSSQL через pyodbc
sql_loader.py           — загрузка данных из DWH (остатки, конкуренты, продажи)
data_validation.py      — валидация и очистка входных DataFrame
feature_engineering.py  — создание 155+ признаков
train.py                — обучение CatBoost + опциональный Optuna
hyperparameter_tuning.py — подбор гиперпараметров через Optuna
model_registry.py       — сохранение/загрузка .cbm + артефактов
predict.py              — инференс (прогноз для последней даты)
batch_predict.py        — пакетный прогноз для диапазона дат
excel_report.py         — генерация Excel-отчётов с форматированием
email_sender.py         — email-уведомления с Excel-вложением

Быстрый старт
-------------
# Обучение (с подбором гиперпараметров):
    python train.py --tune

# Прогноз на последнюю дату в данных:
    python predict.py --data data/raw

# Ежедневный прогноз (сегодня) + Excel-отчёт:
    python batch_predict.py --daily

# Прогноз за период:
    python batch_predict.py --start 2026-01-01 --end 2026-01-31
"""

__version__ = "1.0.0"

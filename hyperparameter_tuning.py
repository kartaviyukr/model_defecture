"""
Подбор гиперпараметров CatBoost через Optuna.

Метрика оптимизации: PR-AUC (Average Precision) на валидационной выборке.
Используется TPE-сэмплер с фиксированным seed для воспроизводимости.

Наилучшие найденные параметры (ноутбук Five, 30 трайлов):
    iterations:        637
    learning_rate:     0.0153
    depth:             10
    l2_leaf_reg:       1.2605
    min_child_samples: 189
    subsample:         0.9474
    colsample_bylevel: 0.7989
    scale_pos_weight:  6.7879
"""

import numpy as np
import optuna
from catboost import CatBoostClassifier
from loguru import logger
from sklearn.metrics import average_precision_score, precision_recall_curve

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_catboost(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials: int = 30,
    imbalance_ratio: float = 7.3,
    random_seed: int = 42,
) -> dict:
    """
    Подбор гиперпараметров CatBoostClassifier через Optuna (TPE).

    Пространство поиска:
    - iterations:        200–1000
    - learning_rate:     0.01–0.1 (log scale)
    - depth:             4–10
    - l2_leaf_reg:       1e-3–10 (log scale)
    - min_child_samples: 10–200
    - subsample:         0.5–1.0
    - colsample_bylevel: 0.5–1.0
    - scale_pos_weight:  1.0–imbalance_ratio

    Args:
        X_train, y_train: Обучающая выборка
        X_val, y_val:     Валидационная выборка (временной сплит)
        n_trials:         Количество попыток Optuna
        imbalance_ratio:  Соотношение классов (кол-во_0 / кол-во_1)
        random_seed:      Seed для воспроизводимости

    Returns:
        Словарь лучших гиперпараметров
    """
    logger.info(
        f"Запуск Optuna: {n_trials} трайлов, "
        f"imbalance_ratio={imbalance_ratio:.2f}, seed={random_seed}"
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 1.0, float(imbalance_ratio)
            ),
            "eval_metric": "AUC",
            "early_stopping_rounds": 50,
            "random_seed": random_seed,
            "verbose": False,
            "allow_writing_files": False,
        }

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

        y_prob = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_prob)
        return pr_auc

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_value = study.best_value

    logger.info(f"Лучший PR-AUC: {best_value:.4f}")
    logger.info(f"Лучшие параметры: {best_params}")

    return best_params


def compute_thresholds(y_true, y_prob) -> dict:
    """
    Вычислить рекомендуемые пороги классификации для трёх стратегий.

    Стратегии:
    - balanced:  максимум F1
    - precision: первый порог с precision >= 0.70
    - recall:    последний порог с recall >= 0.80

    Args:
        y_true: Истинные метки (0/1)
        y_prob: Предсказанные вероятности класса 1

    Returns:
        {"balanced": float, "precision": float, "recall": float}
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # balanced: максимум F1
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    idx_balanced = int(np.argmax(f1_scores))
    thr_balanced = float(thresholds[idx_balanced])

    # precision-optimized: первый порог с precision >= 0.70
    high_prec_mask = precisions[:-1] >= 0.70
    if high_prec_mask.any():
        thr_precision = float(thresholds[high_prec_mask][0])
    else:
        thr_precision = float(thresholds[-1])

    # recall-optimized: последний порог с recall >= 0.80
    high_rec_mask = recalls[:-1] >= 0.80
    if high_rec_mask.any():
        thr_recall = float(thresholds[high_rec_mask][-1])
    else:
        thr_recall = float(thresholds[0])

    result = {
        "balanced": round(thr_balanced, 4),
        "precision": round(thr_precision, 4),
        "recall": round(thr_recall, 4),
    }
    logger.info(f"Пороги классификации: {result}")
    return result

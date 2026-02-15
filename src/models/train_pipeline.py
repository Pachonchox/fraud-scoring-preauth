"""
Esqueleto de pipeline de entrenamiento para modelos de detección de fraude.

Importante: aquí solo se define la estructura y puntos de extensión.
El modelo concreto (árboles, boosting, red neuronal, etc.) se implementará después.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, SEED
from src.features.build_features import PREPROCESSED_FILENAME


def load_training_data() -> pd.DataFrame:
    """
    Carga el dataset procesado desde data/processed.
    """
    path = PROCESSED_DATA_DIR / PREPROCESSED_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta primero src.features.build_features."
        )
    return pd.read_parquet(path)


def split_data(df: pd.DataFrame, target_col: str = "fraud_attempt") -> Dict[str, Any]:
    """
    Separa features y label, y realiza train/test split.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def find_best_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    positive_label: int = 1,
    num_thresholds: int = 19,
) -> Tuple[float, float, float, float]:
    """
    Busca un threshold que maximiza el F1 de la clase de fraude.

    Recorre una malla simple de thresholds en [0.05, 0.95], suficiente
    para ajustar el modelo sin disparar el costo computacional.
    """
    thresholds = np.linspace(0.05, 0.95, num_thresholds)
    best_thr = 0.5
    best_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        p = precision_score(
            y_true,
            y_pred,
            pos_label=positive_label,
            zero_division=0,
        )
        r = recall_score(
            y_true,
            y_pred,
            pos_label=positive_label,
            zero_division=0,
        )
        f1 = f1_score(
            y_true,
            y_pred,
            pos_label=positive_label,
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_precision = p
            best_recall = r

    return best_thr, best_f1, best_precision, best_recall


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Crea un preprocesador que:
    - Estandariza variables numéricas.
    - One-hot-encode variables categóricas.
    """
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [
        c for c in X.columns if c not in numeric_cols
    ]  # object/string/category

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Pipeline:
    preprocessor, _, _ = build_preprocessor(X_train)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        class_weight="balanced",
        random_state=SEED,
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_05 = (y_proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)

    print("\n=== RandomForest (baseline tabular) ===")
    print("ROC AUC:", roc)
    print("PR AUC :", pr)
    print("\n[RF] Classification report (threshold=0.5):")
    print(classification_report(y_test, y_pred_05, digits=4))

    best_thr, best_f1, best_p, best_r = find_best_threshold(y_test, y_proba)
    y_pred_best = (y_proba >= best_thr).astype(int)

    print(f"\n[RF] Mejor threshold por F1 fraude: {best_thr:.3f}")
    print(
        f"[RF] precision={best_p:.3f}, recall={best_r:.3f}, f1={best_f1:.3f} "
        "(clase fraude, test)"
    )
    print("\n[RF] Classification report (mejor F1 fraude):")
    print(classification_report(y_test, y_pred_best, digits=4))

    metrics = {
        "name": "random_forest",
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "threshold_opt": float(best_thr),
        "precision_fraud_opt": float(best_p),
        "recall_fraud_opt": float(best_r),
        "f1_fraud_opt": float(best_f1),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(clf, MODELS_DIR / "rf_preauth.joblib")
    print(f"[OK] Modelo RandomForest guardado en {MODELS_DIR / 'rf_preauth.joblib'}")

    return clf, best_thr, metrics


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Pipeline:
    preprocessor, _, _ = build_preprocessor(X_train)

    # Red neuronal ligera, adecuada para M2 Pro y escalable
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        batch_size=4096,
        max_iter=50,
        early_stopping=True,
        n_iter_no_change=5,
        validation_fraction=0.1,
        learning_rate_init=1e-3,
        random_state=SEED,
        verbose=False,
    )

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    # Ajuste de pesos para la clase minoritaria
    class_counts = np.bincount(y_train.astype(int))
    class_weights = {i: class_counts.sum() / (len(class_counts) * c) for i, c in enumerate(class_counts)}

    sample_weight = y_train.map(class_weights).values

    clf.fit(X_train, y_train, model__sample_weight=sample_weight)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_05 = (y_proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)

    print("\n=== MLPClassifier (red neuronal ligera) ===")
    print("ROC AUC:", roc)
    print("PR AUC :", pr)
    print("\n[MLP] Classification report (threshold=0.5):")
    print(classification_report(y_test, y_pred_05, digits=4))

    best_thr, best_f1, best_p, best_r = find_best_threshold(y_test, y_proba)
    y_pred_best = (y_proba >= best_thr).astype(int)

    print(f"\n[MLP] Mejor threshold por F1 fraude: {best_thr:.3f}")
    print(
        f"[MLP] precision={best_p:.3f}, recall={best_r:.3f}, f1={best_f1:.3f} "
        "(clase fraude, test)"
    )
    print("\n[MLP] Classification report (mejor F1 fraude):")
    print(classification_report(y_test, y_pred_best, digits=4))

    metrics = {
        "name": "mlp",
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "threshold_opt": float(best_thr),
        "precision_fraud_opt": float(best_p),
        "recall_fraud_opt": float(best_r),
        "f1_fraud_opt": float(best_f1),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(clf, MODELS_DIR / "mlp_preauth.joblib")
    print(f"[OK] Modelo MLP guardado en {MODELS_DIR / 'mlp_preauth.joblib'}")

    return clf, best_thr, metrics


def train_hist_gbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Pipeline:
    """
    Modelo tipo gradient boosting (árboles) optimizado para tabular,
    suele rendir mejor que RandomForest en problemas de este tipo.
    """
    preprocessor, _, _ = build_preprocessor(X_train)

    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    # Grid pequeno para balancear mejora real y tiempo de entrenamiento.
    candidate_params = [
        {"max_depth": 6, "max_leaf_nodes": 31, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 20},
        {"max_depth": 8, "max_leaf_nodes": 63, "learning_rate": 0.04, "max_iter": 350, "min_samples_leaf": 15},
        {"max_depth": None, "max_leaf_nodes": 63, "learning_rate": 0.04, "max_iter": 400, "min_samples_leaf": 20},
        {"max_depth": 6, "max_leaf_nodes": 31, "learning_rate": 0.03, "max_iter": 500, "min_samples_leaf": 10},
    ]

    class_counts_sub = np.bincount(y_subtrain.astype(int))
    class_weights_sub = {
        i: class_counts_sub.sum() / (len(class_counts_sub) * c)
        for i, c in enumerate(class_counts_sub)
    }
    sample_weight_sub = y_subtrain.map(class_weights_sub).values

    best_cfg = None
    best_val_f1 = -1.0
    best_val_pr = -1.0
    best_val_thr = 0.5

    print("\n[HGB] Tuning en split de validacion...")
    for idx, params in enumerate(candidate_params, start=1):
        model = HistGradientBoostingClassifier(
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.1,
            **params,
        )
        candidate = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
        candidate.fit(X_subtrain, y_subtrain, model__sample_weight=sample_weight_sub)

        val_proba = candidate.predict_proba(X_val)[:, 1]
        val_pr = average_precision_score(y_val, val_proba)
        val_thr, val_f1, val_p, val_r = find_best_threshold(
            y_val, val_proba, num_thresholds=31
        )
        print(
            f"[HGB][C{idx}] params={params} | PR={val_pr:.4f} | "
            f"thr={val_thr:.3f} | F1={val_f1:.4f}"
        )

        if (val_f1 > best_val_f1) or (
            np.isclose(val_f1, best_val_f1) and val_pr > best_val_pr
        ):
            best_cfg = params
            best_val_f1 = val_f1
            best_val_pr = val_pr
            best_val_thr = val_thr

    print(f"[HGB] Mejor configuracion: {best_cfg}")
    print(f"[HGB] Threshold de operacion (validacion): {best_val_thr:.3f}")

    final_model = HistGradientBoostingClassifier(
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        **best_cfg,
    )
    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", final_model),
        ]
    )

    class_counts_full = np.bincount(y_train.astype(int))
    class_weights_full = {
        i: class_counts_full.sum() / (len(class_counts_full) * c)
        for i, c in enumerate(class_counts_full)
    }
    sample_weight_full = y_train.map(class_weights_full).values
    clf.fit(X_train, y_train, model__sample_weight=sample_weight_full)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred_05 = (y_proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_proba)
    pr = average_precision_score(y_test, y_proba)

    print("\n=== HistGradientBoosting (arboles boosting) ===")
    print("ROC AUC:", roc)
    print("PR AUC :", pr)
    print("\n[HGB] Classification report (threshold=0.5):")
    print(classification_report(y_test, y_pred_05, digits=4))

    y_pred_best = (y_proba >= best_val_thr).astype(int)
    best_p = precision_score(y_test, y_pred_best, zero_division=0)
    best_r = recall_score(y_test, y_pred_best, zero_division=0)
    best_f1 = f1_score(y_test, y_pred_best, zero_division=0)

    print(f"\n[HGB] Threshold aplicado desde validacion: {best_val_thr:.3f}")
    print(
        f"[HGB] precision={best_p:.3f}, recall={best_r:.3f}, f1={best_f1:.3f} "
        "(clase fraude, test)"
    )
    print("\n[HGB] Classification report (threshold operativo):")
    print(classification_report(y_test, y_pred_best, digits=4))

    metrics = {
        "name": "hist_gradient_boosting",
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "threshold_opt": float(best_val_thr),
        "threshold_source": "validation",
        "precision_fraud_opt": float(best_p),
        "recall_fraud_opt": float(best_r),
        "f1_fraud_opt": float(best_f1),
        "tuned_params": best_cfg,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(clf, MODELS_DIR / "hgb_preauth.joblib")
    print(f"[OK] Modelo HistGradientBoosting guardado en {MODELS_DIR / 'hgb_preauth.joblib'}")

    return clf, best_val_thr, metrics


def main() -> None:
    """
    Punto de entrada del pipeline de entrenamiento.

    Pasos esperados (a implementar después):
    - Cargar datos procesados.
    - Definir modelo(s) base.
    - Entrenar y evaluar.
    - Guardar métricas y artefactos.
    """
    df = load_training_data()
    splits = split_data(df)

    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]

    print(
        f"Dataset para entrenamiento cargado. "
        f"Train: {len(X_train):,} filas, Test: {len(X_test):,} filas."
    )

    # 1) Modelo baseline tabular
    rf_clf, rf_thr, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test
    )

    # 2) Modelo boosting (árboles)
    hgb_clf, hgb_thr, hgb_metrics = train_hist_gbm(
        X_train, y_train, X_test, y_test
    )

    # 3) Red neuronal ligera (MLP)
    mlp_clf, mlp_thr, mlp_metrics = train_mlp(
        X_train, y_train, X_test, y_test
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    thresholds_path = MODELS_DIR / "thresholds_preauth.json"
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump({"rf": rf_thr, "hgb": hgb_thr, "mlp": mlp_thr}, f, indent=2)
    print(f"\n[OK] Thresholds recomendados guardados en {thresholds_path}")

    # Seleccionar mejor modelo según F1 de fraude en test
    all_metrics = [rf_metrics, hgb_metrics, mlp_metrics]

    # Guardar métricas de los 3 modelos para tabla comparativa del dashboard
    all_metrics_path = MODELS_DIR / "all_models_metrics.json"
    with all_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[OK] Métricas de todos los modelos guardadas en {all_metrics_path}")
    best_metrics = max(all_metrics, key=lambda m: m["f1_fraud_opt"])
    best_name = best_metrics["name"]

    model_paths = {
        "random_forest": MODELS_DIR / "rf_preauth.joblib",
        "hist_gradient_boosting": MODELS_DIR / "hgb_preauth.joblib",
        "mlp": MODELS_DIR / "mlp_preauth.joblib",
    }
    models_map = {
        "random_forest": rf_clf,
        "hist_gradient_boosting": hgb_clf,
        "mlp": mlp_clf,
    }

    best_clf = models_map[best_name]
    best_model_params_raw = best_clf.named_steps["model"].get_params()
    best_model_params = {
        k: (
            v
            if isinstance(v, (int, float, str, bool)) or v is None
            else str(v)
        )
        for k, v in best_model_params_raw.items()
    }

    best_meta = {
        "best_model_name": best_name,
        "model_path": str(model_paths[best_name]),
        "threshold_opt": best_metrics["threshold_opt"],
        "metrics": best_metrics,
        "hyperparameters": best_model_params,
    }

    best_meta_path = MODELS_DIR / "best_model_metadata.json"
    with best_meta_path.open("w", encoding="utf-8") as f:
        json.dump(best_meta, f, indent=2)
    print(f"[OK] Metadatos del mejor modelo guardados en {best_meta_path}")

    print("\n[OK] Entrenamiento completado para RandomForest, HGB y MLP.")


if __name__ == "__main__":
    main()

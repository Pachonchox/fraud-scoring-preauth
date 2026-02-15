"""
Ejemplo mínimo de cómo probar el mejor modelo de fraude pre-autorización
desde Python (modo script o REPL).
"""

from __future__ import annotations

import json

import pandas as pd
from joblib import load

from src.config import PROCESSED_DATA_DIR
from src.features.build_features import (
    PREPROCESSED_FILENAME,
    _add_business_features,
    _drop_leaky_and_id_columns,
)


def main(n_samples: int = 20) -> None:
    parquet_path = PROCESSED_DATA_DIR / PREPROCESSED_FILENAME
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        # Fallback si hay incompatibilidades de pyarrow en el entorno.
        raw_csv = PROCESSED_DATA_DIR.parent / "raw" / "ml_dataset.csv"
        df_raw = pd.read_csv(raw_csv)
        df = _drop_leaky_and_id_columns(_add_business_features(df_raw))
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)
    X = df.drop(columns=["fraud_attempt"])
    y = df["fraud_attempt"]

    # Tomamos un pequeño sample de transacciones
    sample = X.sample(n_samples, random_state=0)
    y_true = y.loc[sample.index]

    # Cargar mejor modelo y threshold recomendado
    with open("models/best_model_metadata.json", "r", encoding="utf-8") as f:
        best_meta = json.load(f)

    model_path = best_meta["model_path"]
    thr = best_meta["threshold_opt"]
    best_name = best_meta["best_model_name"]

    model = load(model_path)

    proba = model.predict_proba(sample)[:, 1]
    pred = (proba >= thr).astype(int)

    out = sample.copy()
    out["proba_fraude"] = proba
    out["pred_fraude"] = pred
    out["label_real"] = y_true.values

    print(f"Mejor modelo: {best_name}")
    print(f"Threshold recomendado: {thr:.3f}")
    print(out[["proba_fraude", "pred_fraude", "label_real"]].head(n_samples))


if __name__ == "__main__":
    main()

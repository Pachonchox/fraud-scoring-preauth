"""
Módulo placeholder para construcción de features adicionales.

Aquí puedes agregar transformaciones sobre `ml_dataset.csv` y guardar
un dataset procesado en `data/processed/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


PREPROCESSED_FILENAME = "ml_dataset_preauth.parquet"


def _add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features adicionales de negocio, siempre pre-autorización.
    """
    if "channel" in df.columns:
        df["ecommerce_flag"] = (df["channel"] == "ecommerce").astype("int8")

    if "hour" in df.columns:
        df["night_txn_flag"] = df["hour"].between(0, 5).astype("int8")

    # Escala log para montos
    if "amount" in df.columns:
        df["log_amount"] = np.log1p(df["amount"].clip(lower=0))
        df["high_amount_flag"] = (df["amount"] > 220_000).astype("int8")

    # Comercios y países
    if {"merchant_country", "channel", "ip_country"}.issubset(df.columns):
        df["foreign_merchant_flag"] = (df["merchant_country"] != "CL").astype("int8")
        df["ecom_foreign_ip_flag"] = (
            (df["channel"] == "ecommerce")
            & (df["ip_country"].notna())
            & (df["ip_country"] != "CL")
            & (df["ip_country"] != "none")
        ).astype("int8")
        df["ecom_ip_merchant_mismatch_flag"] = (
            (df["channel"] == "ecommerce")
            & (df["ip_country"].notna())
            & (df["ip_country"] != "none")
            & (df["merchant_country"].notna())
            & (df["merchant_country"] != df["ip_country"])
        ).astype("int8")

    if "pos_entry_mode" in df.columns:
        df["risky_entry_mode_flag"] = df["pos_entry_mode"].isin(
            ["manual", "magstripe"]
        ).astype("int8")

    if "cvv_result" in df.columns:
        df["cvv_negative_flag"] = (df["cvv_result"] == "N").astype("int8")

    if "three_ds_status" in df.columns:
        df["three_ds_failed_flag"] = (df["three_ds_status"] == "failed").astype(
            "int8"
        )

    if "card_txn_count_1h" in df.columns:
        df["card_velocity_high_1h_flag"] = (df["card_txn_count_1h"] >= 3).astype("int8")

    if "device_txn_count_1h" in df.columns:
        df["device_velocity_high_1h_flag"] = (
            df["device_txn_count_1h"] >= 5
        ).astype("int8")

    if "card_pct_declines_prev30" in df.columns:
        df["card_decline_spike_flag"] = (df["card_pct_declines_prev30"] >= 0.40).astype(
            "int8"
        )

    return df


def _drop_leaky_and_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas que no deben usarse en un score pre-autorización:
    - Códigos de respuesta / autorización.
    - Labels y metadatos de fraude.
    - Identificadores de alta cardinalidad.
    - Timestamps crudos (dejamos solo dow/hour/month).
    """
    cols_to_drop = [
        # IDs transaccionales y de entidades
        "txn_id",
        "card_id",
        "merchant_id",
        "device_id",
        "issuer_id",
        "acquirer_id",
        "terminal_id",
        "rrn",
        "auth_code",
        "pan_bin",
        # Campos de decisión / post-decision
        "response_code",
        # Labels de fraude y metadatos (excepto el target fraud_attempt)
        "fraud_scenario",
        "label_source",
        "days_to_confirm",
        # Timestamps completos (dejamos solo features de tiempo derivadas)
        "transmission_ts_utc",
        "local_txn_date",
        "local_txn_time",
    ]

    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)

    # Asegurar tipo entero para el target
    if "fraud_attempt" in df.columns:
        df["fraud_attempt"] = df["fraud_attempt"].astype("int8")

    return df


def build_base_features() -> None:
    """
    Construye un dataset base para modelado pre-autorización y lo guarda en data/processed.

    No hace encoding ni escalado todavía; eso se maneja en el pipeline de modelos.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    input_path = RAW_DATA_DIR / "ml_dataset.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"No se encontró {input_path}. Genera primero los datos sintéticos."
        )

    df = pd.read_csv(input_path)

    df = _add_business_features(df)
    df = _drop_leaky_and_id_columns(df)

    output_path = PROCESSED_DATA_DIR / PREPROCESSED_FILENAME
    df.to_parquet(output_path, index=False)
    print(f"[OK] Dataset pre-autorización guardado en {output_path}")


if __name__ == "__main__":
    build_base_features()

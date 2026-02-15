from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

from src.config import PROCESSED_DATA_DIR
from src.features.build_features import (
    PREPROCESSED_FILENAME,
    _add_business_features,
    _drop_leaky_and_id_columns,
)

APP_VERSION = "v3.0 — portfolio narrativo"

# ---------------------------------------------------------------------------
# Presets de escenarios para la demo
# ---------------------------------------------------------------------------
SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "Compra normal": {
        "amount": 22000.0,
        "channel": "pos",
        "pos_entry_mode": "chip",
        "merchant_country": "CL",
        "mcc": "5411",
        "ip_country": "none",
        "three_ds_status": "none",
        "cvv_result": "U",
    },
    "Ecommerce sospechoso": {
        "amount": 180000.0,
        "channel": "ecommerce",
        "pos_entry_mode": "ecommerce",
        "merchant_country": "CL",
        "mcc": "5732",
        "ip_country": "US",
        "three_ds_status": "failed",
        "cvv_result": "N",
    },
    "Card testing": {
        "amount": 1500.0,
        "channel": "ecommerce",
        "pos_entry_mode": "ecommerce",
        "merchant_country": "CL",
        "mcc": "5999",
        "ip_country": "MX",
        "three_ds_status": "failed",
        "cvv_result": "N",
    },
    "Compra internacional": {
        "amount": 90000.0,
        "channel": "ecommerce",
        "pos_entry_mode": "ecommerce",
        "merchant_country": "US",
        "mcc": "5812",
        "ip_country": "CL",
        "three_ds_status": "authenticated",
        "cvv_result": "M",
    },
    "Fraude — CNP takeover": {
        "amount": 280000.0,
        "channel": "ecommerce",
        "pos_entry_mode": "ecommerce",
        "merchant_country": "CL",
        "mcc": "5732",
        "ip_country": "ES",
        "three_ds_status": "failed",
        "cvv_result": "N",
    },
    "Fraude — Counterfeit POS": {
        "amount": 145000.0,
        "channel": "pos",
        "pos_entry_mode": "magstripe",
        "merchant_country": "CL",
        "mcc": "5999",
        "ip_country": "none",
        "three_ds_status": "none",
        "cvv_result": "U",
    },
}

SCENARIO_EXPLANATIONS: Dict[str, str] = {
    "Compra normal": "Bajo riesgo esperado: compra presencial típica, monto cotidiano y señales estables.",
    "Ecommerce sospechoso": "Riesgo medio/alto: ecommerce con CVV negativo y 3DS fallido.",
    "Card testing": "Patrón de pruebas de tarjeta: montos pequeños, CVV negativo y 3DS fallido.",
    "Compra internacional": "Riesgo bajo/medio: compra ecommerce transfronteriza con IP local y 3DS autenticado.",
    "Fraude — CNP takeover": "Toma de cuenta: IP extranjera, 3DS fallido y monto alto.",
    "Fraude — Counterfeit POS": "Tarjeta falsificada en POS: magstripe/manual y consumo elevado.",
}

FRAUD_PATTERN_DESCRIPTIONS: Dict[str, str] = {
    "card_testing": "Ráfagas de montos pequeños en ecommerce, CVV negativo y dispositivos maliciosos. Objetivo: validar datos robados antes de un fraude mayor.",
    "cnp_takeover": "Uso de dispositivo nuevo y/o IP extranjera con escalamiento de montos. El atacante obtiene credenciales y opera desde otra ubicación.",
    "counterfeit_cp": "Uso presencial con magstripe o manual, concentrado en comercios de alto riesgo. Tarjetas clonadas desde datos de banda magnética.",
    "friendly_fraud": "Operación aparentemente legítima que luego se desconoce vía chargeback. El tarjetahabiente real disputa la transacción.",
}

FIELD_DESCRIPTIONS: Dict[str, str] = {
    "amount": "Monto de la transacción en CLP.",
    "log_amount": "Transformación logarítmica del monto para estabilizar escala.",
    "high_amount_flag": "Marca montos altos sobre umbral operativo.",
    "channel": "Canal de la operación: pos/ecommerce.",
    "card_present": "Indica si la tarjeta estuvo presente físicamente.",
    "pos_entry_mode": "Modo de captura: chip, contactless, magstripe, manual o ecommerce.",
    "mcc": "Categoría del comercio (Merchant Category Code).",
    "merchant_country": "País del comercio.",
    "ip_country": "País de IP de origen en ecommerce.",
    "eci": "Indicador de comercio electrónico.",
    "three_ds_status": "Estado de autenticación 3DS.",
    "cvv_result": "Resultado de validación CVV.",
    "avs_result": "Resultado de validación de dirección.",
    "tokenized": "Indica si la transacción usó token.",
    "wallet_type": "Tipo de billetera digital si aplica.",
    "dow": "Día de semana de la transacción.",
    "hour": "Hora de la transacción.",
    "month": "Mes de la transacción.",
    "ecommerce_flag": "Bandera de operación ecommerce.",
    "night_txn_flag": "Bandera de operación nocturna (00-05h).",
    "foreign_merchant_flag": "Bandera de comercio fuera de Chile.",
    "ecom_foreign_ip_flag": "Bandera de ecommerce con IP extranjera.",
    "ecom_ip_merchant_mismatch_flag": "Bandera de mismatch entre país IP y país del comercio.",
    "risky_entry_mode_flag": "Bandera de modo riesgoso (manual o magstripe).",
    "cvv_negative_flag": "Bandera de CVV negativo.",
    "three_ds_failed_flag": "Bandera de 3DS fallido.",
    "card_velocity_high_1h_flag": "Bandera de alta velocidad de transacciones por tarjeta en 1h.",
    "device_velocity_high_1h_flag": "Bandera de alta velocidad por dispositivo en 1h.",
    "card_decline_spike_flag": "Bandera de aumento de rechazos recientes por tarjeta.",
    "card_txn_count_1h": "Número de transacciones previas de la tarjeta en 1 hora.",
    "card_txn_count_24h": "Número de transacciones previas de la tarjeta en 24 horas.",
    "card_sum_amount_24h": "Suma de montos previos de la tarjeta en 24 horas.",
    "card_unique_merchants_24h": "Cantidad de comercios únicos usados por la tarjeta en 24 horas.",
    "card_time_since_prev_min": "Minutos desde la transacción previa de la tarjeta.",
    "card_pct_declines_prev30": "Porcentaje de rechazos en las últimas 30 transacciones de la tarjeta.",
    "device_txn_count_1h": "Número de transacciones previas del dispositivo en 1 hora.",
    "device_txn_count_24h": "Número de transacciones previas del dispositivo en 24 horas.",
    "device_sum_amount_24h": "Suma de montos previos del dispositivo en 24 horas.",
    "device_unique_merchants_24h": "Cantidad de comercios únicos usados por el dispositivo en 24 horas.",
    "device_time_since_prev_min": "Minutos desde la transacción previa del dispositivo.",
    "device_pct_declines_prev30": "Porcentaje de rechazos en las últimas 30 transacciones del dispositivo.",
    "card_new_merchant_24h_flag": "Bandera de comercio nuevo para la tarjeta en 24h.",
    "device_new_merchant_24h_flag": "Bandera de comercio nuevo para el dispositivo en 24h.",
}

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

@st.cache_data
def load_dataset() -> pd.DataFrame:
    parquet_path = PROCESSED_DATA_DIR / PREPROCESSED_FILENAME
    try:
        return pd.read_parquet(parquet_path)
    except Exception:
        raw_csv = PROCESSED_DATA_DIR.parent / "raw" / "ml_dataset.csv"
        df_raw = pd.read_csv(raw_csv)
        df_proc = _drop_leaky_and_id_columns(_add_business_features(df_raw))
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df_proc.to_parquet(parquet_path, index=False)
        return df_proc


@st.cache_data
def load_dataset_profile() -> Dict[str, Any]:
    profile_path = PROCESSED_DATA_DIR.parent / "raw" / "dataset_profile.json"
    with profile_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_best_model() -> Dict[str, Any]:
    with open("models/best_model_metadata.json", "r", encoding="utf-8") as f:
        best_meta = json.load(f)
    model_path = Path(best_meta["model_path"])
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path
    return {
        "model": load(model_path),
        "threshold": float(best_meta["threshold_opt"]),
        "meta": best_meta,
    }


@st.cache_data
def load_all_models_metrics() -> list[Dict[str, Any]]:
    metrics_path = Path("models/all_models_metrics.json")
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ---------------------------------------------------------------------------
# Estilos CSS
# ---------------------------------------------------------------------------

def apply_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; padding-bottom: 2.2rem; }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: #64748b !important;
        }
        div[data-testid="stMetricValue"] div {
            color: #1e293b !important;
            font-weight: 700 !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 2px solid #e2e8f0;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #1d4ed8 !important;
            font-weight: 700 !important;
        }

        /* Hero card */
        .hero-card {
            background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 50%, #2563eb 100%);
            border-radius: 12px;
            padding: 22px 28px;
            color: #ffffff;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(30,64,175,0.25);
        }
        .hero-card h2, .hero-card h3, .hero-card p { color: #ffffff !important; margin: 0; }
        .hero-card p { opacity: 0.92; margin-top: 6px; font-size: 1.05rem; }

        /* Pattern cards */
        .pattern-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 16px 18px;
            height: 100%;
        }
        .pattern-card h4 { margin: 0 0 6px 0; color: #1e293b; font-size: 0.95rem; }
        .pattern-card p { margin: 0; color: #475569; font-size: 0.85rem; line-height: 1.45; }

        /* Info badge */
        .info-badge {
            display: inline-block;
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
            border-radius: 20px;
            padding: 6px 16px;
            font-weight: 600;
            font-size: 0.88rem;
            margin: 8px 0;
        }

        /* Highlight box */
        .highlight-box {
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            border-radius: 0 8px 8px 0;
            padding: 12px 16px;
            margin: 10px 0;
            color: #1e40af;
            font-size: 0.9rem;
        }

        /* Use-case cards */
        .usecase-card {
            background: #f1f5f9;
            border-radius: 10px;
            padding: 16px 18px;
            height: 100%;
        }
        .usecase-card h4 { margin: 0 0 6px 0; color: #1e293b; font-size: 0.93rem; }
        .usecase-card p { margin: 0; color: #475569; font-size: 0.85rem; line-height: 1.45; }

        /* Score bar */
        .score-bar-container {
            background: #e2e8f0;
            border-radius: 8px;
            height: 28px;
            position: relative;
            overflow: hidden;
            margin: 8px 0;
        }
        .score-bar-fill {
            height: 100%;
            border-radius: 8px;
            transition: width 0.4s ease;
        }
        .score-bar-label {
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-weight: 700;
            font-size: 0.85rem;
            color: #1e293b;
        }

        /* Savings highlight */
        .savings-metric {
            background: #ecfdf5;
            border: 2px solid #10b981;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }
        .savings-metric .number { font-size: 1.6rem; font-weight: 800; color: #065f46; }
        .savings-metric .label { font-size: 0.82rem; color: #047857; text-transform: uppercase; letter-spacing: 0.03em; }

        /* Winner badge */
        .winner-badge {
            display: inline-block;
            background: #fef3c7;
            color: #92400e;
            border-radius: 4px;
            padding: 2px 8px;
            font-weight: 700;
            font-size: 0.78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Funciones utilitarias
# ---------------------------------------------------------------------------

def build_default_row(df: pd.DataFrame) -> pd.Series:
    """Selecciona una fila base no-fraudulenta con historial normal (no primera transacción)."""
    candidates = df[
        (df["fraud_attempt"] == 0)
        & (df.get("device_time_since_prev_min", pd.Series(dtype=float)).lt(500000) if "device_time_since_prev_min" in df.columns else True)
        & (df.get("card_time_since_prev_min", pd.Series(dtype=float)).lt(500000) if "card_time_since_prev_min" in df.columns else True)
    ]
    if candidates.empty:
        candidates = df[df["fraud_attempt"] == 0]
    return candidates.drop(columns=["fraud_attempt"]).sample(1, random_state=42).iloc[0].copy()


def apply_scenario(base_row: pd.Series, scenario: str) -> pd.Series:
    row = base_row.copy()
    preset = SCENARIO_PRESETS.get(scenario, {})
    for key, value in preset.items():
        if key in row.index:
            row[key] = value

    channel = str(row.get("channel", "pos"))
    amount = float(row.get("amount", 0.0))
    is_ecom = channel == "ecommerce"

    # Derived features coherentes con el canal
    if "log_amount" in row.index:
        row["log_amount"] = np.log1p(max(amount, 0.0))
    if "high_amount_flag" in row.index:
        row["high_amount_flag"] = int(amount > 220000)
    if "card_present" in row.index:
        row["card_present"] = int(not is_ecom)
    if "ecommerce_flag" in row.index:
        row["ecommerce_flag"] = int(is_ecom)
    if "foreign_merchant_flag" in row.index:
        row["foreign_merchant_flag"] = int(row.get("merchant_country", "CL") != "CL")
    if "ecom_foreign_ip_flag" in row.index:
        row["ecom_foreign_ip_flag"] = int(
            is_ecom and row.get("ip_country", "none") not in ("CL", "none")
        )
    if "ecom_ip_merchant_mismatch_flag" in row.index:
        row["ecom_ip_merchant_mismatch_flag"] = int(
            is_ecom
            and row.get("ip_country", "none") not in ("none", row.get("merchant_country", "CL"))
        )
    # POS no tiene 3DS ni IP
    if not is_ecom:
        if "three_ds_status" in row.index:
            row["three_ds_status"] = "none"
        if "ip_country" in row.index:
            row["ip_country"] = "none"
        if "eci" in row.index:
            row["eci"] = "none"
        if "tokenized" in row.index:
            row["tokenized"] = 0
        if "wallet_type" in row.index:
            row["wallet_type"] = "none"

    # Forzar señales de riesgo diferenciadas por tipo de fraude
    is_fraud_scenario = scenario.startswith("Fraude") or scenario == "Card testing"

    if is_fraud_scenario and is_ecom:
        # --- Fraude ecommerce (CNP takeover, card testing) ---
        # Señales clave: 3DS fallido + CVV negativo + IP extranjera + dispositivo nuevo
        if "three_ds_failed_flag" in row.index:
            row["three_ds_failed_flag"] = 1
        if "cvv_negative_flag" in row.index:
            row["cvv_negative_flag"] = 1
        if "ecom_foreign_ip_flag" in row.index:
            row["ecom_foreign_ip_flag"] = 1
        # Dispositivo nuevo (atacante usa device desconocido)
        if "device_time_since_prev_min" in row.index:
            row["device_time_since_prev_min"] = 999999.0
        if "device_new_merchant_24h_flag" in row.index:
            row["device_new_merchant_24h_flag"] = 1
        if "device_txn_count_1h" in row.index:
            row["device_txn_count_1h"] = 0.0
        if "device_txn_count_24h" in row.index:
            row["device_txn_count_24h"] = 0.0
        if "device_sum_amount_24h" in row.index:
            row["device_sum_amount_24h"] = 0.0

    elif is_fraud_scenario and not is_ecom:
        # --- Fraude POS (counterfeit CP) ---
        # Señales: magstripe + comercio caliente + dispositivo nuevo
        if "risky_entry_mode_flag" in row.index:
            row["risky_entry_mode_flag"] = 1
        if "is_hot_merchant" in row.index:
            row["is_hot_merchant"] = 1
        # Dispositivo nuevo: sin historial previo (clonación)
        if "device_txn_count_1h" in row.index:
            row["device_txn_count_1h"] = 0.0
        if "device_txn_count_24h" in row.index:
            row["device_txn_count_24h"] = 0.0
        if "device_sum_amount_24h" in row.index:
            row["device_sum_amount_24h"] = 0.0
        if "device_time_since_prev_min" in row.index:
            row["device_time_since_prev_min"] = 999999.0
        if "device_new_merchant_24h_flag" in row.index:
            row["device_new_merchant_24h_flag"] = 1
        # 3DS no aplica a POS
        if "three_ds_failed_flag" in row.index:
            row["three_ds_failed_flag"] = 0
        if "cvv_negative_flag" in row.index:
            row["cvv_negative_flag"] = 0

    return row


def explain_risk_from_features(
    row: pd.Series, score: float, threshold: float, scenario_hint: str,
) -> list[str]:
    reasons: list[str] = []

    if score >= threshold:
        reasons.append(f"Score ({score:.2%}) sobre el threshold operativo ({threshold:.2f}).")
    else:
        reasons.append(f"Score ({score:.2%}) bajo el threshold operativo ({threshold:.2f}).")

    is_ecom = row.get("channel", "") == "ecommerce"
    if is_ecom:
        reasons.append("Operación ecommerce: mayor superficie de riesgo que POS.")
    if row.get("cvv_result", "") == "N" or (int(row.get("cvv_negative_flag", 0)) == 1 and is_ecom):
        reasons.append("CVV negativo detectado.")
    if (row.get("three_ds_status", "") == "failed" or int(row.get("three_ds_failed_flag", 0)) == 1) and is_ecom:
        reasons.append("3DS fallido — señal fuerte de fraude CNP.")
    if row.get("ip_country", "none") not in ("CL", "none") or int(row.get("ecom_foreign_ip_flag", 0)) == 1:
        reasons.append("IP extranjera para tarjeta de emisor chileno.")
    if int(row.get("risky_entry_mode_flag", 0)) == 1 or row.get("pos_entry_mode", "") in ("manual", "magstripe"):
        reasons.append("Modo de ingreso riesgoso (manual/magstripe).")
    if int(row.get("is_hot_merchant", 0)) == 1:
        reasons.append("Comercio con alta concentración de fraude histórico (hot merchant).")
    if float(row.get("amount", 0.0)) >= 220000 or int(row.get("high_amount_flag", 0)) == 1:
        reasons.append("Monto alto sobre umbral operativo.")
    if int(row.get("card_velocity_high_1h_flag", 0)) == 1 or float(row.get("card_txn_count_1h", 0)) >= 3:
        reasons.append("Alta velocidad de transacciones por tarjeta (1h).")
    if int(row.get("device_velocity_high_1h_flag", 0)) == 1 or float(row.get("device_txn_count_1h", 0)) >= 5:
        reasons.append("Alta velocidad de transacciones por dispositivo (1h).")
    if int(row.get("card_decline_spike_flag", 0)) == 1 or float(row.get("card_pct_declines_prev30", 0.0)) >= 0.4:
        reasons.append("Spike de rechazos recientes en la tarjeta.")
    if scenario_hint in FRAUD_PATTERN_DESCRIPTIONS:
        reasons.append(f"Patrón: {FRAUD_PATTERN_DESCRIPTIONS[scenario_hint]}")

    if len(reasons) <= 1:
        reasons.append("Sin señales fuertes de fraude; comportamiento consistente con transacción regular.")

    return reasons


def get_threshold_curve(df: pd.DataFrame, model: Any) -> pd.DataFrame:
    if "threshold_curve" in st.session_state:
        return st.session_state["threshold_curve"]

    with st.spinner("Calculando curva precisión/recall..."):
        eval_df = df.sample(min(len(df), 120000), random_state=42)
        X_eval = eval_df.drop(columns=["fraud_attempt"])
        y_eval = eval_df["fraud_attempt"].astype(int).values
        probs = model.predict_proba(X_eval)[:, 1]

        rows = []
        for thr in np.round(np.linspace(0.10, 0.99, 90), 3):
            pred = probs >= thr
            tp = int(np.sum((pred == 1) & (y_eval == 1)))
            fp = int(np.sum((pred == 1) & (y_eval == 0)))
            fn = int(np.sum((pred == 0) & (y_eval == 1)))
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            rows.append({"threshold": float(thr), "precision": prec, "recall": rec, "f1": f1})

        curve = pd.DataFrame(rows)
        st.session_state["threshold_curve"] = curve
        return curve


def metrics_at_threshold(curve: pd.DataFrame, threshold: float) -> Dict[str, float]:
    idx = (curve["threshold"] - threshold).abs().idxmin()
    row = curve.loc[idx]
    return {"precision": float(row["precision"]), "recall": float(row["recall"]),
            "f1": float(row["f1"]), "threshold": float(row["threshold"])}


def compute_business_impact(
    base_rate: float, precision: float, recall: float,
    total_tx: int, avg_ticket: float,
    cost_per_fn_factor: float, cost_per_fp_factor: float,
) -> Tuple[float, float, float, float, float, float]:
    fraud_tx = total_tx * base_rate
    tp = recall * fraud_tx
    fn = fraud_tx - tp
    fp = tp * (1 / max(precision, 1e-8) - 1)
    cost_fn = fn * avg_ticket * cost_per_fn_factor
    cost_fp = fp * avg_ticket * cost_per_fp_factor
    total_cost = cost_fn + cost_fp
    return tp, fp, fn, cost_fn, cost_fp, total_cost


# ---------------------------------------------------------------------------
# Gráficos Altair
# ---------------------------------------------------------------------------

def render_light_bar_chart(data: pd.DataFrame, x_col: str, y_col: str, y_format: str = ".2%") -> None:
    chart = (
        alt.Chart(data)
        .mark_bar(color="#0f4c81", cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X(f"{x_col}:N", sort="-y", title=x_col),
            y=alt.Y(f"{y_col}:Q", title=y_col),
            tooltip=[
                alt.Tooltip(f"{x_col}:N", title=x_col),
                alt.Tooltip(f"{y_col}:Q", title=y_col, format=y_format),
            ],
        )
        .properties(height=280)
        .configure_axis(labelColor="#0f172a", titleColor="#0f172a", gridColor="#e2e8f0")
        .configure_view(strokeOpacity=0)
        .configure(background="#ffffff")
    )
    st.altair_chart(chart, use_container_width=True)


def render_light_line_chart(data: pd.DataFrame) -> None:
    melted = data.melt(id_vars="threshold", value_vars=["precision", "recall", "f1"])
    chart = (
        alt.Chart(melted)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("threshold:Q", title="threshold"),
            y=alt.Y("value:Q", title="métrica"),
            color=alt.Color("variable:N", scale=alt.Scale(range=["#0f4c81", "#0ea5e9", "#14b8a6"])),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".2f"),
                alt.Tooltip("variable:N"),
                alt.Tooltip("value:Q", format=".3f"),
            ],
        )
        .properties(height=320)
        .configure_axis(labelColor="#0f172a", titleColor="#0f172a", gridColor="#e2e8f0")
        .configure_view(strokeOpacity=0)
        .configure(background="#ffffff")
    )
    st.altair_chart(chart, use_container_width=True)


def describe_field(field_name: str) -> str:
    if field_name in FIELD_DESCRIPTIONS:
        return FIELD_DESCRIPTIONS[field_name]
    if field_name.endswith("_flag"):
        return "Bandera binaria de señal de riesgo."
    if "count" in field_name:
        return "Conteo histórico de intensidad/velocidad."
    if "sum_amount" in field_name:
        return "Acumulado histórico de monto."
    if "pct_declines" in field_name:
        return "Tasa histórica de rechazo."
    if "time_since" in field_name:
        return "Tiempo desde el evento previo."
    return "Variable pre-autorización usada por el modelo."


# ---------------------------------------------------------------------------
# Renderizado de tabs
# ---------------------------------------------------------------------------

def render_hero(meta: Dict[str, Any], model_metrics: Dict[str, Any]) -> None:
    st.markdown(
        """
        <div class="hero-card">
          <h2 style="font-size:1.5rem;">Fraud Scoring Pre-autorización</h2>
          <p>Sistema de machine learning para detectar fraude antes de que la transacción sea
          autorizada — emisor Mastercard Chile.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Algoritmo", "HistGradientBoosting")
    c2.metric("Capacidad de detección", f"{model_metrics['roc_auc']:.1%}")
    c3.metric("Precisión en alertas", f"{model_metrics['precision_fraud_opt']:.1%}")
    c4.metric("Fraudes detectados", f"{model_metrics['recall_fraud_opt']:.1%}")
    c5.metric("Umbral de decisión", f"{meta['threshold_opt']:.0%}")


def render_tab_problema(profile: Dict[str, Any]) -> None:
    # --- El problema ---
    st.markdown("### El problema")
    st.markdown(
        """
Cada vez que un tarjetahabiente pasa su tarjeta o compra en línea, el emisor tiene **milisegundos**
para decidir si aprueba o rechaza. Un modelo de scoring pre-autorización evalúa la probabilidad
de fraude usando solo la información del mensaje ISO 8583, *antes* de enviar la respuesta.
"""
    )

    fraud_profile = profile.get("fraud", {})
    data_shape = profile.get("dataset_shape", {})
    channel_mix = profile.get("channel_mix", {})
    ecom_pct = channel_mix.get("ecommerce", 0)
    pos_pct = channel_mix.get("pos", 0)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Transacciones", f"{int(data_shape.get('rows', 0)):,}")
    d2.metric("Tasa de fraude", f"{float(fraud_profile.get('fraud_rate', 0)):.2%}")
    d3.metric("Ecommerce", f"{ecom_pct:.0%}")
    d4.metric("POS presencial", f"{pos_pct:.0%}")

    # --- Distribución de canales ---
    st.markdown("### Distribución de canales")
    mix_df = pd.DataFrame({"canal": list(channel_mix.keys()), "proporción": list(channel_mix.values())})
    if not mix_df.empty:
        render_light_bar_chart(mix_df, "canal", "proporción")

    issuer_info = profile.get("issuer_profile", {})
    date_range = profile.get("date_range_utc", {})
    st.caption(
        f"Emisor: {issuer_info.get('issuer_name', 'N/A')} ({issuer_info.get('issuer_country', 'CL')}) · "
        f"Período: {date_range.get('start', '')[:10]} a {date_range.get('end', '')[:10]}"
    )

    # --- Patrones de fraude ---
    st.markdown("### Los 4 patrones de fraude")

    pattern_names = {
        "card_testing": "Card Testing",
        "cnp_takeover": "CNP Takeover",
        "counterfeit_cp": "Counterfeit CP",
        "friendly_fraud": "Friendly Fraud",
    }

    cols = st.columns(2)
    for i, (key, desc) in enumerate(FRAUD_PATTERN_DESCRIPTIONS.items()):
        with cols[i % 2]:
            st.markdown(
                f'<div class="pattern-card"><h4>{pattern_names.get(key, key)}</h4>'
                f'<p>{desc}</p></div>',
                unsafe_allow_html=True,
            )
    st.write("")  # spacing

    scenario_mix = fraud_profile.get("fraud_scenarios", {})
    if scenario_mix:
        scenario_df = pd.DataFrame(
            {"patrón": list(scenario_mix.keys()), "proporción": list(scenario_mix.values())}
        )
        render_light_bar_chart(scenario_df, "patrón", "proporción")

    # --- Decisiones de diseño ---
    st.markdown("### Decisiones de diseño")
    st.markdown(
        """
- **Emisor único**: simula el caso real de un banco construyendo su propio modelo de scoring. En producción, cada emisor tiene distribuciones distintas.
- **Sin ATM**: el caso de uso es scoring de compras (*purchase*). Los retiros tienen controles distintos (PIN, límites diarios).
- **Ecommerce mayoritario (58%)**: refleja la realidad post-2020 en Chile. Mastercard reporta >55% de transacciones digitales en emisores grandes de LATAM.
- **Fraude 0.8%**: tasa típica de *intentos* para un emisor mediano. La tasa confirmada suele ser ~0.3-0.5%.
- **4 escenarios**: cubren los vectores principales — validación de datos robados, toma de cuenta, clonación física y disputa fraudulenta.
"""
    )


def render_tab_modelo(
    meta: Dict[str, Any],
    all_metrics: list[Dict[str, Any]],
    df: pd.DataFrame,
) -> None:
    feature_columns = df.drop(columns=["fraud_attempt"]).columns.tolist()
    model_metrics = meta["metrics"]

    # --- Feature engineering ---
    st.markdown("### Feature engineering")
    st.markdown(
        f'<div class="info-badge">{len(feature_columns)} features · 0 post-autorización · 0 leakage</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="highlight-box"><strong>Principio anti-leakage</strong>: solo se usan features '
        'disponibles en el mensaje de autorización ISO 8583. Ningún dato post-decisión '
        '(response_code, auth_code) entra al modelo.</div>',
        unsafe_allow_html=True,
    )

    feat_table = pd.DataFrame([
        {"Tipo": "Transaccionales", "Cantidad": 5, "Ejemplos": "amount, log_amount, channel, mcc, card_present"},
        {"Tipo": "Autenticación", "Cantidad": 5, "Ejemplos": "cvv_result, three_ds_status, eci, tokenized, wallet_type"},
        {"Tipo": "Geográficas", "Cantidad": 5, "Ejemplos": "merchant_country, ip_country, foreign flags, mismatch"},
        {"Tipo": "Temporales", "Cantidad": 4, "Ejemplos": "dow, hour, month, night_txn_flag"},
        {"Tipo": "Velocidad tarjeta", "Cantidad": 7, "Ejemplos": "card_txn_count_1h/24h, card_sum_amount_24h, decline_spike"},
        {"Tipo": "Velocidad dispositivo", "Cantidad": 7, "Ejemplos": "device_txn_count_1h/24h, device_sum_amount_24h"},
        {"Tipo": "Flags de riesgo", "Cantidad": 5, "Ejemplos": "risky_entry_mode, cvv_negative, ecom_foreign_ip"},
    ])
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

    with st.expander("Ver diccionario completo de features"):
        rows = [{"Campo": col, "Descripción": describe_field(col)} for col in sorted(feature_columns)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Comparación de modelos ---
    st.markdown("### Comparación de modelos")

    model_name_map = {
        "random_forest": "RandomForest",
        "hist_gradient_boosting": "HistGradientBoosting",
        "mlp": "MLPClassifier",
    }
    best_name = meta["best_model_name"]

    if all_metrics:
        comp_rows = []
        for m in all_metrics:
            name = m["name"]
            comp_rows.append({
                "Modelo": model_name_map.get(name, name),
                "ROC AUC": f"{m['roc_auc']:.4f}",
                "PR AUC": f"{m['pr_auc']:.4f}",
                "F1 fraude": f"{m['f1_fraud_opt']:.4f}",
                "Precisión": f"{m['precision_fraud_opt']:.4f}",
                "Recall": f"{m['recall_fraud_opt']:.4f}",
                "Ganador": "✓" if name == best_name else "",
            })
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            f"""
| Modelo | ROC AUC | PR AUC | F1 fraude | Precisión | Recall |
|--------|---------|--------|-----------|-----------|--------|
| **HistGradientBoosting** | **{model_metrics['roc_auc']:.4f}** | **{model_metrics['pr_auc']:.4f}** | **{model_metrics['f1_fraud_opt']:.4f}** | **{model_metrics['precision_fraud_opt']:.4f}** | **{model_metrics['recall_fraud_opt']:.4f}** |
| RandomForest | — | — | — | — | — |
| MLPClassifier | — | — | — | — | — |
"""
        )

    # --- Métricas del ganador ---
    st.markdown("### Métricas del modelo ganador")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC AUC", f"{model_metrics['roc_auc']:.4f}")
    m2.metric("PR AUC", f"{model_metrics['pr_auc']:.4f}")
    m3.metric("F1 fraude", f"{model_metrics['f1_fraud_opt']:.4f}")
    m4.metric("Precisión", f"{model_metrics['precision_fraud_opt']:.2%}")
    m5.metric("Recall", f"{model_metrics['recall_fraud_opt']:.2%}")

    # --- Por qué HGB ---
    st.markdown("### ¿Por qué HistGradientBoosting?")
    st.markdown(
        """
- **Categóricos nativos**: maneja features categóricos sin one-hot encoding, evitando explosión de dimensionalidad.
- **Early stopping**: detiene el entrenamiento cuando la validación deja de mejorar (patience=10).
- **Eficiente en CPU**: ideal para M2 Pro (~2 min de entrenamiento), escalable a producción.
- **Regularización implícita**: via max_depth, max_leaf_nodes y min_samples_leaf.
"""
    )

    tuned = meta.get("hyperparameters", {})
    hp_table = pd.DataFrame([
        {"Parámetro": "learning_rate", "Valor": str(tuned.get("learning_rate", "N/A"))},
        {"Parámetro": "max_iter", "Valor": str(tuned.get("max_iter", "N/A"))},
        {"Parámetro": "max_depth", "Valor": str(tuned.get("max_depth", "N/A"))},
        {"Parámetro": "max_leaf_nodes", "Valor": str(tuned.get("max_leaf_nodes", "N/A"))},
        {"Parámetro": "min_samples_leaf", "Valor": str(tuned.get("min_samples_leaf", "N/A"))},
        {"Parámetro": "early_stopping", "Valor": "True (patience=10)"},
        {"Parámetro": "split", "Valor": "80% train / 20% test, estratificado"},
    ])
    st.dataframe(hp_table, use_container_width=True, hide_index=True)

    # --- Stack tecnológico ---
    st.markdown("### Stack tecnológico")
    stack_table = pd.DataFrame([
        {"Componente": "Lenguaje", "Tecnología": "Python 3.11+"},
        {"Componente": "ML", "Tecnología": "scikit-learn (HGB, RF, MLP)"},
        {"Componente": "Datos", "Tecnología": "pandas + pyarrow (Parquet)"},
        {"Componente": "Visualización", "Tecnología": "Streamlit + Altair"},
        {"Componente": "Serialización", "Tecnología": "joblib"},
        {"Componente": "Ejecución", "Tecnología": "Apple M2 Pro (~3 min gen + ~2 min train)"},
    ])
    st.dataframe(stack_table, use_container_width=True, hide_index=True)
    st.caption(
        "Escalable: NumPy vectorizado, rolling windows pandas (migrable a Spark/Polars), "
        "modelo exportable a ONNX. Pipeline reproducible (seed=42)."
    )


def render_tab_impacto(
    df: pd.DataFrame,
    model: Any,
    threshold_default: float,
    base_rate: float,
) -> None:
    st.markdown("### Impacto económico del modelo")
    st.markdown(
        "Ajusta los parámetros del negocio para estimar el ahorro que genera el modelo "
        "comparado con no tener detección automatizada."
    )

    curve = get_threshold_curve(df, model)

    p1, p2, p3 = st.columns(3)
    total_tx = p1.slider("Volumen de transacciones", 10000, 500000, 100000, 10000)
    avg_ticket = p2.slider("Ticket promedio (CLP)", 5000.0, 300000.0, 25000.0, 5000.0)
    manual_threshold = p3.slider("Threshold operativo", 0.10, 0.99, float(threshold_default), 0.01)

    cost_fn_factor = 1.0
    cost_fp_factor = 0.1

    at_thr = metrics_at_threshold(curve, manual_threshold)
    precision = at_thr["precision"]
    recall = at_thr["recall"]
    f1 = at_thr["f1"]

    tp, fp, fn, cost_fn, cost_fp, total_cost = compute_business_impact(
        base_rate=base_rate, precision=precision, recall=recall,
        total_tx=total_tx, avg_ticket=avg_ticket,
        cost_per_fn_factor=cost_fn_factor, cost_per_fp_factor=cost_fp_factor,
    )
    baseline_cost = total_tx * base_rate * avg_ticket * cost_fn_factor
    savings = baseline_cost - total_cost

    # Resultados
    col_left, col_right = st.columns([2, 1])

    with col_left:
        r1, r2, r3 = st.columns(3)
        r1.metric("Precisión fraude", f"{precision:.2%}")
        r2.metric("Recall fraude", f"{recall:.2%}")
        r3.metric("F1 fraude", f"{f1:.2%}")

        r4, r5, r6 = st.columns(3)
        r4.metric("Fraudes detectados (TP)", f"{tp:,.0f}")
        r5.metric("Fraudes perdidos (FN)", f"{fn:,.0f}")
        r6.metric("Falsos positivos (FP)", f"{fp:,.0f}")

    with col_right:
        st.markdown(
            f'<div class="savings-metric">'
            f'<div class="label">Ahorro estimado</div>'
            f'<div class="number">${savings:,.0f}</div>'
            f'<div class="label">CLP</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.metric("Costo sin modelo", f"${baseline_cost:,.0f} CLP")
        st.metric("Costo con modelo", f"${total_cost:,.0f} CLP")
        st.metric("Costo por fricción (FP)", f"${cost_fp:,.0f} CLP")

    # Curva
    st.markdown("### Curva de desempeño por threshold")
    chart_df = curve[["threshold", "precision", "recall", "f1"]].copy()
    render_light_line_chart(chart_df)
    st.caption(
        f"El threshold {threshold_default:.2f} fue calibrado en validación para maximizar F1. "
        "En producción se ajusta según la política de riesgo del emisor y puede diferenciarse por segmento."
    )

    # Casos de uso
    st.markdown("### Casos de uso en producción")
    uc1, uc2, uc3 = st.columns(3)
    with uc1:
        st.markdown(
            '<div class="usecase-card"><h4>Autorización en tiempo real</h4>'
            '<p>El score se calcula en milisegundos durante el flujo de autorización. '
            'Transacciones sobre el threshold se envían a revisión o se declinan automáticamente.</p></div>',
            unsafe_allow_html=True,
        )
    with uc2:
        st.markdown(
            '<div class="usecase-card"><h4>Cola de revisión manual</h4>'
            '<p>El equipo de fraude prioriza su cola según el score. '
            'Los analistas revisan primero las transacciones de mayor riesgo, optimizando su tiempo.</p></div>',
            unsafe_allow_html=True,
        )
    with uc3:
        st.markdown(
            '<div class="usecase-card"><h4>Ajuste de políticas</h4>'
            '<p>El comité de riesgo usa la curva precision/recall para calibrar el threshold '
            'por segmento (canal, MCC, país) según su apetito de riesgo.</p></div>',
            unsafe_allow_html=True,
        )


def render_tab_demo(
    df: pd.DataFrame,
    model: Any,
    threshold_default: float,
) -> None:
    st.markdown("### El modelo en acción")
    st.markdown("Selecciona un escenario y observa cómo el modelo asigna el score de riesgo.")

    base_row = build_default_row(df)

    scenario = st.radio(
        "Escenario",
        list(SCENARIO_PRESETS.keys()),
        horizontal=True,
        key="demo_scenario",
    )

    row = apply_scenario(base_row, scenario)
    X_input = pd.DataFrame([row])
    score = float(model.predict_proba(X_input)[:, 1][0])

    # Determinar hint de fraude
    fraud_hint = "none"
    if scenario == "Card testing":
        fraud_hint = "card_testing"
    elif "CNP" in scenario:
        fraud_hint = "cnp_takeover"
    elif "Counterfeit" in scenario:
        fraud_hint = "counterfeit_cp"

    risk_reasons = explain_risk_from_features(
        row=X_input.iloc[0], score=score, threshold=threshold_default, scenario_hint=fraud_hint,
    )

    # Layout de resultado
    col_score, col_context = st.columns([2, 1])

    with col_score:
        c1, c2, c3 = st.columns(3)
        c1.metric("Score de fraude", f"{score:.2%}")
        c2.metric("Threshold", f"{threshold_default:.2f}")

        decision = "Revisar / Declinar" if score >= threshold_default else "Aprobar"
        c3.metric("Decisión", decision)

        score_pct = min(score * 100, 100)
        bar_width = max(score_pct, 5)
        if score_pct < 30:
            bar_color = "#16a34a"
        elif score_pct < 60:
            bar_color = "#d97706"
        else:
            bar_color = "#dc2626"

        st.markdown(
            f'<div class="score-bar-container">'
            f'<div class="score-bar-fill" style="width:{bar_width}%;background:{bar_color};"></div>'
            f'<div class="score-bar-label">{score:.2%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_context:
        st.markdown("**Señales de riesgo detectadas:**")
        for reason in risk_reasons:
            st.markdown(f"- {reason}")

    # Explicación del escenario
    st.markdown("---")
    st.markdown(f"**{scenario}**: {SCENARIO_EXPLANATIONS.get(scenario, '')}")

    # Detalles del preset
    preset = SCENARIO_PRESETS.get(scenario, {})
    if preset:
        detail_df = pd.DataFrame([{"Campo": k, "Valor": str(v)} for k, v in preset.items()])
        with st.expander("Ver datos de la transacción"):
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.caption(
        "El modelo no reemplaza al analista: prioriza qué transacciones revisar primero. "
        "Score alto = mayor probabilidad relativa de fraude, no certeza."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Fraud Scoring — Portfolio", layout="wide")
    apply_style()

    df = load_dataset()
    profile = load_dataset_profile()
    model_info = load_best_model()
    model = model_info["model"]
    threshold_default = model_info["threshold"]
    meta = model_info["meta"]
    model_metrics = meta["metrics"]
    base_rate = float(df["fraud_attempt"].mean())
    all_metrics = load_all_models_metrics()

    render_hero(meta, model_metrics)
    st.caption(f"Versión: {APP_VERSION}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["El problema", "El modelo", "Impacto", "Demo"]
    )

    with tab1:
        render_tab_problema(profile)

    with tab2:
        render_tab_modelo(meta, all_metrics, df)

    with tab3:
        render_tab_impacto(df, model, threshold_default, base_rate)

    with tab4:
        render_tab_demo(df, model, threshold_default)


if __name__ == "__main__":
    main()

"""
Generación de dataset sintético de fraude transaccional.

Este módulo implementa el script que genera:
- auth_events.csv
- labels.csv
- ml_dataset.csv

Los archivos se escriben en data/raw/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import string

from src.config import RAW_DATA_DIR, SEED


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # CONFIG
    # =========================================================
    rng = np.random.default_rng(SEED)

    N_ROWS = 300_000

    # Único emisor en Chile
    ISSUER_ID = "BANK_CL_UNIQUE_001"
    ISSUER_NAME = "Banco Unico Chile"
    HOME_COUNTRY = "CL"

    # Mix realista: ecommerce domina post-pandemia en emisores chilenos (Mastercard LATAM 2025)
    POS_SHARE = 0.42
    ECOM_SHARE = 0.58

    # Fraude total (intentos)
    FRAUD_RATE = 0.008  # ~0.8% de intentos

    # Tamaños de entidades
    N_CARDS = 45_000
    N_MERCHANTS = 20_000
    N_DEVICES = 65_000  # devices “normales” persistentes
    N_FRAUD_DEVICES = 800  # devices “maliciosos” (bots, etc.)

    START_UTC = datetime(2025, 11, 1)
    END_UTC = datetime(2026, 2, 13)

    COUNTRIES = ["CL", "AR", "PE", "CO", "BR", "MX", "US", "ES"]
    MERCH_COUNTRY_P = [0.58, 0.06, 0.05, 0.05, 0.05, 0.05, 0.10, 0.06]

    CURRENCIES = ["CLP", "USD"]

    # MCC: (prob, median_CLP, sigma) — sin ATM (6011), con MCCs digitales
    MCCS = {
        "5411": (0.17, 18000, 0.60),  # grocery (supermercados, Lider/Jumbo online)
        "5812": (0.14, 12000, 0.70),  # restaurants (delivery apps, iFood, Rappi)
        "5541": (0.07, 26000, 0.50),  # fuel (solo POS presencial)
        "5999": (0.07, 38000, 0.80),  # misc retail (marketplaces, MercadoLibre)
        "4111": (0.05, 8500, 0.50),  # transit (metro, buses, Uber)
        "4814": (0.08, 9000, 0.60),  # telecom (recargas, planes)
        "5732": (0.07, 65000, 0.90),  # electronics (retail tech, Falabella)
        "5311": (0.06, 45000, 0.70),  # dept store (tiendas por depto)
        "7399": (0.05, 30000, 0.90),  # biz services (SaaS, consulting)
        "5651": (0.07, 42000, 0.70),  # apparel (moda online, Shein, Zara)
        "7995": (0.05, 25000, 0.80),  # entertainment (gaming, ticketing)
        "5816": (0.06, 8000, 0.50),  # digital goods (apps, gaming, in-app)
        "5815": (0.06, 5500, 0.40),  # streaming / suscripciones (Netflix, Spotify)
    }
    mcc_keys = np.array(list(MCCS.keys()))
    mcc_probs = np.array([MCCS[m][0] for m in mcc_keys])
    mcc_probs = mcc_probs / mcc_probs.sum()

    # =========================================================
    # HELPERS
    # =========================================================
    def rand_id(prefix, n=12):
        alphabet = string.ascii_uppercase + string.digits
        return prefix + "_" + "".join(rng.choice(list(alphabet), size=n))

    def sample_time_uniform(n):
        span = (END_UTC - START_UTC).total_seconds()
        secs = rng.uniform(0, span, size=n)
        return np.array([START_UTC + timedelta(seconds=float(s)) for s in secs], dtype="datetime64[ns]")

    def lognormal_amount(median, sigma):
        mu = np.log(max(median, 1.0))
        val = rng.lognormal(mean=mu, sigma=sigma)
        return float(max(100, np.round(val, 0)))

    def weighted_choice(values, probs, size):
        p = np.array(probs, dtype=float)
        p = p / p.sum()
        return rng.choice(values, size=size, p=p)

    def response_from_risk(risk):
        # ISO-ish
        base_decline = 0.015
        p_decline = min(0.92, base_decline + 0.85 * risk)
        if rng.random() < p_decline:
            # “05” domina cuando riesgo alto
            return rng.choice(["05", "51", "54", "14", "57"], p=[0.58, 0.20, 0.08, 0.07, 0.07])
        return "00"

    def dow_hour_month(ts_utc):
        s = pd.to_datetime(ts_utc)
        return s.dt.dayofweek.astype(int), s.dt.hour.astype(int), s.dt.month.astype(int)

    # =========================================================
    # 1) ENTIDADES (cards, merchants, devices)
    # =========================================================
    # Cards (mismo emisor)
    card_ids = np.array([rand_id("CARD", 14) for _ in range(N_CARDS)])
    bins = weighted_choice(["548012", "520082", "531234", "222300", "535678"], [0.28, 0.22, 0.20, 0.18, 0.12], N_CARDS)
    product_type = weighted_choice(["credit", "debit", "prepaid"], [0.65, 0.30, 0.05], N_CARDS)
    cards = pd.DataFrame(
        {
            "card_id": card_ids,
            "issuer_id": ISSUER_ID,
            "issuer_name": ISSUER_NAME,
            "home_country": HOME_COUNTRY,
            "pan_bin": bins,
            "product_type": product_type,
        }
    )

    # Devices persistentes “normales” y “fraud devices” (bots)
    device_ids = np.array([rand_id("DEV", 14) for _ in range(N_DEVICES)])
    fraud_device_ids = np.array([rand_id("FDEV", 12) for _ in range(N_FRAUD_DEVICES)])

    # Cada tarjeta tiene 0–2 devices “habituales” (para ecom)
    # (muchas tarjetas casi nunca hacen ecom; igual les damos posibilidad)
    card_primary_dev = rng.choice(
        np.append(device_ids, ["none"]),
        size=N_CARDS,
        p=np.append(np.full(N_DEVICES, 0.97 / N_DEVICES), [0.03]),
    )
    card_secondary_dev = rng.choice(
        np.append(device_ids, ["none"]),
        size=N_CARDS,
        p=np.append(np.full(N_DEVICES, 0.985 / N_DEVICES), [0.015]),
    )

    # Merchants
    merchant_ids = np.array([rand_id("M", 10) for _ in range(N_MERCHANTS)])
    terminal_ids = np.array([rand_id("T", 8) for _ in range(N_MERCHANTS)])
    acquirer_id = weighted_choice(
        ["ACQ_CL_001", "ACQ_CL_002", "ACQ_INT_001", "ACQ_INT_002"],
        [0.46, 0.30, 0.14, 0.10],
        N_MERCHANTS,
    )
    mcc_for_merchant = rng.choice(mcc_keys, size=N_MERCHANTS, p=mcc_probs)
    merchant_country = rng.choice(COUNTRIES, size=N_MERCHANTS, p=MERCH_COUNTRY_P)
    merchant_city = weighted_choice(
        [
            "Santiago",
            "Valparaiso",
            "Concepcion",
            "Antofagasta",
            "Buenos Aires",
            "Lima",
            "Bogota",
            "Sao Paulo",
            "CDMX",
            "Miami",
            "Madrid",
        ],
        [0.29, 0.06, 0.06, 0.05, 0.09, 0.06, 0.06, 0.08, 0.07, 0.10, 0.08],
        N_MERCHANTS,
    )
    merchant_name = weighted_choice(
        [
            "SUPERMERCADO",
            "RESTAURANTE",
            "BENCINERA",
            "TIENDA",
            "ELECTRONICA",
            "TRANSPORTE",
            "TELECOM",
            "SERVICIOS",
            "ENTRETENCION",
            "FARMACIA",
            "STREAMING",
            "DELIVERY",
        ],
        [0.15, 0.13, 0.07, 0.10, 0.07, 0.05, 0.07, 0.09, 0.07, 0.07, 0.06, 0.07],
        N_MERCHANTS,
    ) + " " + weighted_choice(
        ["NORTE", "SUR", "CENTRO", "PLUS", "MAX", "EXPRESS", "UNO", "DOS", "TRES"],
        [0.12, 0.12, 0.12, 0.12, 0.12, 0.16, 0.08, 0.08, 0.08],
        N_MERCHANTS,
    )

    merchants = pd.DataFrame(
        {
            "merchant_id": merchant_ids,
            "terminal_id": terminal_ids,
            "acquirer_id": acquirer_id,
            "mcc": mcc_for_merchant,
            "merchant_country": merchant_country,
            "merchant_city": merchant_city,
            "merchant_name": merchant_name,
        }
    )

    # “Hot merchants” (concentran fraude)
    hot_merchants_idx = rng.choice(np.arange(N_MERCHANTS), size=max(50, N_MERCHANTS // 200), replace=False)
    merchants["is_hot_merchant"] = 0
    merchants.loc[hot_merchants_idx, "is_hot_merchant"] = 1

    # Índice merchants por MCC para coherencia
    merchants_by_mcc = {m: merchants[merchants["mcc"] == m].index.values for m in mcc_keys}

    # =========================================================
    # 2) GENERAR TRANSACCIONES "NORMALES" (baseline)
    #    (Luego reemplazamos una parte por campañas de fraude)
    # =========================================================
    # Base: selecciona card + mcc + merchant consistente
    card_idx = rng.integers(0, N_CARDS, size=N_ROWS)
    mcc = rng.choice(mcc_keys, size=N_ROWS, p=mcc_probs)

    merchant_row_idx = np.empty(N_ROWS, dtype=int)
    for mm in mcc_keys:
        mask = mcc == mm
        candidates = merchants_by_mcc[mm]
        merchant_row_idx[mask] = rng.choice(candidates, size=mask.sum())

    ts = sample_time_uniform(N_ROWS)

    auth = pd.DataFrame(
        {
            "txn_id": [rand_id("TXN", 16) for _ in range(N_ROWS)],
            "mti": weighted_choice(["0100", "0200"], [0.92, 0.08], N_ROWS),
            "stan": rng.integers(1, 999999, size=N_ROWS),
            "rrn": [rand_id("RRN", 12) for _ in range(N_ROWS)],
            "transmission_ts_utc": ts,
        }
    )

    auth = auth.join(cards.iloc[card_idx].reset_index(drop=True))
    auth = auth.join(merchants.iloc[merchant_row_idx].reset_index(drop=True), rsuffix="_m")

    # Guardrail: todas las tarjetas pertenecen a un emisor chileno unico.
    if auth["issuer_id"].nunique() != 1 or auth["home_country"].nunique() != 1:
        raise ValueError("Se detecto mas de un emisor/home_country en el dataset.")
    if auth["issuer_id"].iloc[0] != ISSUER_ID or auth["home_country"].iloc[0] != "CL":
        raise ValueError("El emisor/home_country no coincide con la configuracion unica de Chile.")

    # Channel mix — solo POS y ecommerce (sin ATM, caso de uso es scoring de compras)
    channel = weighted_choice(["pos", "ecommerce"], [POS_SHARE, ECOM_SHARE], N_ROWS)
    auth["channel"] = channel
    auth["card_present"] = (auth["channel"] == "pos").astype(int)

    # ---------------------------------------------------------------
    # Entry mode coherente por canal (estándares Mastercard 2025)
    # ---------------------------------------------------------------
    is_pos = auth["channel"].values == "pos"
    is_ecom = auth["channel"].values == "ecommerce"

    # POS: chip domina, contactless en fuerte crecimiento en Chile
    pos_entry = weighted_choice(
        ["chip", "contactless", "magstripe", "manual"],
        [0.55, 0.38, 0.05, 0.02],
        N_ROWS,
    )
    # Ecommerce: compra web estándar + COF (recurrentes) + in-app
    ecom_entry = weighted_choice(
        ["ecommerce", "credential_on_file", "in_app"],
        [0.72, 0.18, 0.10],
        N_ROWS,
    )
    auth["pos_entry_mode"] = np.where(is_pos, pos_entry, ecom_entry)

    # ---------------------------------------------------------------
    # Señales de autenticación ecommerce (Mastercard Identity Check / 3DS 2.0)
    # ---------------------------------------------------------------
    auth["pos_condition_code"] = np.where(is_ecom, "59", "00")

    # ECI: varía según entry mode ecommerce
    eci_standard = weighted_choice(["05", "06", "07"], [0.60, 0.26, 0.14], N_ROWS)
    eci_cof = weighted_choice(["05", "06", "07"], [0.72, 0.20, 0.08], N_ROWS)  # COF tiene más auth
    eci_inapp = weighted_choice(["05", "06"], [0.80, 0.20], N_ROWS)  # in-app mejor autenticado
    auth["eci"] = np.where(
        is_ecom,
        np.where(
            auth["pos_entry_mode"].values == "credential_on_file",
            eci_cof,
            np.where(auth["pos_entry_mode"].values == "in_app", eci_inapp, eci_standard),
        ),
        "none",
    )

    # 3DS: tasas altas de autenticación (3DS 2.0 mandatorio, Mastercard SCA)
    auth["three_ds_status"] = np.where(
        is_ecom,
        weighted_choice(
            ["authenticated", "attempted", "failed", "not_enrolled", "challenge"],
            [0.74, 0.10, 0.05, 0.07, 0.04],
            N_ROWS,
        ),
        "none",
    )

    # CVV: diferenciado por canal y entry mode (realismo Mastercard)
    # POS chip/contactless: iCVV validado por chip → alta tasa de match
    # POS magstripe: CVV1 en pista → menor confiabilidad
    # POS manual: sin chip, sin pista → resultado variable
    cvv_chip = weighted_choice(["M", "U"], [0.85, 0.15], N_ROWS)
    cvv_magstripe = weighted_choice(["M", "N", "U"], [0.70, 0.10, 0.20], N_ROWS)
    cvv_manual = weighted_choice(["M", "N", "U"], [0.15, 0.25, 0.60], N_ROWS)
    cvv_ecom = weighted_choice(["M", "N", "U"], [0.91, 0.06, 0.03], N_ROWS)

    pos_cvv = np.where(
        auth["pos_entry_mode"].values == "magstripe",
        cvv_magstripe,
        np.where(auth["pos_entry_mode"].values == "manual", cvv_manual, cvv_chip),
    )
    auth["cvv_result"] = np.where(is_ecom, cvv_ecom, np.where(is_pos, pos_cvv, "U"))

    # AVS: solo relevante en ecommerce
    auth["avs_result"] = np.where(
        is_ecom,
        weighted_choice(["Y", "N", "U"], [0.62, 0.22, 0.16], N_ROWS),
        "U",
    )

    # Tokenización ecommerce (~32%, Apple Pay y Google Pay en Chile)
    auth["tokenized"] = np.where(
        is_ecom,
        rng.choice([0, 1], size=N_ROWS, p=[0.68, 0.32]),
        0,
    )
    auth["wallet_type"] = np.where(
        auth["tokenized"].values == 1,
        weighted_choice(["apple_pay", "google_pay"], [0.48, 0.52], N_ROWS),
        "none",
    )

    # Device ID persistente por tarjeta (solo ecommerce)
    dev_for_card = np.where(
        rng.random(N_ROWS) < 0.85,
        card_primary_dev[card_idx],
        card_secondary_dev[card_idx],
    )
    auth["device_id"] = np.where(is_ecom, dev_for_card, "none")

    # IP country: ecommerce, mayoritariamente CL
    auth["ip_country"] = np.where(
        is_ecom,
        rng.choice(
            COUNTRIES,
            size=N_ROWS,
            p=[0.82, 0.03, 0.02, 0.02, 0.03, 0.02, 0.04, 0.02],
        ),
        "none",
    )

    # Moneda
    auth["currency"] = np.where(
        auth["merchant_country"].values == "CL",
        "CLP",
        rng.choice(CURRENCIES, size=N_ROWS, p=[0.65, 0.35]),
    )

    # Amount por MCC
    amounts = np.zeros(N_ROWS, dtype=float)
    for mm in mcc_keys:
        mask = auth["mcc"].values == mm
        median, sigma = MCCS[mm][1], MCCS[mm][2]
        n = mask.sum()
        if n:
            amounts[mask] = [lognormal_amount(median, sigma) for _ in range(n)]
    auth["amount"] = amounts

    # Time features
    auth["dow"], auth["hour"], auth["month"] = dow_hour_month(auth["transmission_ts_utc"])
    ts_dt = pd.to_datetime(auth["transmission_ts_utc"])
    auth["local_txn_date"] = ts_dt.dt.date
    auth["local_txn_time"] = ts_dt.dt.time

    # =========================================================
    # 3) LABELS + CAMPAÑAS DE FRAUDE (realistas)
    # =========================================================
    labels = pd.DataFrame({"txn_id": auth["txn_id"].values})
    labels["fraud_attempt"] = 0
    labels["fraud_scenario"] = "none"
    labels["label_source"] = "none"
    labels["days_to_confirm"] = np.nan

    # Elegimos qué filas serán fraude (pero luego las “moldeamos” por patrón)
    fraud_mask = rng.random(N_ROWS) < FRAUD_RATE
    labels.loc[fraud_mask, "fraud_attempt"] = 1

    # Distribución equilibrada POS/Ecom
    scenarios = rng.choice(
        ["card_testing", "cnp_takeover", "counterfeit_cp", "friendly_fraud"],
        size=fraud_mask.sum(),
        p=[0.26, 0.30, 0.24, 0.20],
    )
    labels.loc[fraud_mask, "fraud_scenario"] = scenarios
    labels.loc[fraud_mask, "label_source"] = rng.choice(
        ["case_confirmed", "chargeback", "intel_list"],
        size=fraud_mask.sum(),
        p=[0.45, 0.40, 0.15],
    )
    labels.loc[fraud_mask, "days_to_confirm"] = rng.integers(3, 60, size=fraud_mask.sum())

    # --- 3.1 Card testing campaign
    mask_ct = fraud_mask.copy()
    mask_ct[fraud_mask] = scenarios == "card_testing"

    auth.loc[mask_ct, "channel"] = "ecommerce"
    auth.loc[mask_ct, "card_present"] = 0
    auth.loc[mask_ct, "pos_entry_mode"] = "ecommerce"
    auth.loc[mask_ct, "device_id"] = rng.choice(fraud_device_ids, size=mask_ct.sum(), replace=True)
    auth.loc[mask_ct, "ip_country"] = rng.choice(
        ["CL", "US", "BR", "MX", "ES"],
        size=mask_ct.sum(),
        p=[0.35, 0.20, 0.15, 0.15, 0.15],
    )
    auth.loc[mask_ct, "amount"] = rng.integers(100, 3000, size=mask_ct.sum()).astype(float)
    auth.loc[mask_ct, "cvv_result"] = rng.choice(["N", "U"], size=mask_ct.sum(), p=[0.88, 0.12])
    auth.loc[mask_ct, "three_ds_status"] = rng.choice(
        ["failed", "not_enrolled"],
        size=mask_ct.sum(),
        p=[0.70, 0.30],
    )
    auth.loc[mask_ct, "eci"] = rng.choice(["06", "07"], size=mask_ct.sum(), p=[0.35, 0.65])
    auth.loc[mask_ct, "tokenized"] = 0
    auth.loc[mask_ct, "wallet_type"] = "none"

    ct_idx = np.where(mask_ct)[0]
    if len(ct_idx) > 0:
        burst_centers = sample_time_uniform(max(20, len(ct_idx) // 800))
        assigned = rng.integers(0, len(burst_centers), size=len(ct_idx))
        jit_min = rng.normal(loc=0, scale=12, size=len(ct_idx))
        new_ts = np.array(
            [
                (
                    pd.Timestamp(burst_centers[assigned[i]])
                    + pd.Timedelta(minutes=float(jit_min[i]))
                ).to_datetime64()
                for i in range(len(ct_idx))
            ],
            dtype="datetime64[ns]",
        )
        auth.loc[ct_idx, "transmission_ts_utc"] = new_ts

    # --- 3.2 CNP takeover
    mask_cnp = fraud_mask.copy()
    mask_cnp[fraud_mask] = scenarios == "cnp_takeover"

    auth.loc[mask_cnp, "channel"] = "ecommerce"
    auth.loc[mask_cnp, "card_present"] = 0
    auth.loc[mask_cnp, "pos_entry_mode"] = "ecommerce"

    auth.loc[mask_cnp, "device_id"] = rng.choice(
        np.append(device_ids, fraud_device_ids),
        size=mask_cnp.sum(),
        p=np.append(
            np.full(N_DEVICES, 0.75 / N_DEVICES),
            np.full(N_FRAUD_DEVICES, 0.25 / N_FRAUD_DEVICES),
        ),
    )
    auth.loc[mask_cnp, "ip_country"] = rng.choice(
        [c for c in COUNTRIES if c != "CL"],
        size=mask_cnp.sum(),
        p=[0.18, 0.12, 0.12, 0.12, 0.12, 0.20, 0.14],
    )
    auth.loc[mask_cnp, "cvv_result"] = rng.choice(
        ["M", "N", "U"],
        size=mask_cnp.sum(),
        p=[0.52, 0.38, 0.10],
    )
    auth.loc[mask_cnp, "three_ds_status"] = rng.choice(
        ["authenticated", "attempted", "failed", "not_enrolled"],
        size=mask_cnp.sum(),
        p=[0.18, 0.18, 0.35, 0.29],
    )
    auth.loc[mask_cnp, "eci"] = rng.choice(
        ["05", "06", "07"],
        size=mask_cnp.sum(),
        p=[0.35, 0.35, 0.30],
    )

    cnp_idx = np.where(mask_cnp)[0]
    if len(cnp_idx) > 0:
        big = rng.random(len(cnp_idx)) < 0.35
        auth.loc[cnp_idx[big], "amount"] *= rng.uniform(1.5, 3.5, size=big.sum())
        auth.loc[cnp_idx[big], "amount"] = np.clip(
            auth.loc[cnp_idx[big], "amount"],
            15000,
            450000,
        )

        tmp = auth.loc[cnp_idx, ["card_id"]].copy()
        starts = sample_time_uniform(max(50, tmp["card_id"].nunique()))
        start_map = {cid: starts[i] for i, cid in enumerate(tmp["card_id"].unique())}
        jit_min = rng.normal(loc=40, scale=50, size=len(cnp_idx))
        new_ts = np.array(
            [
                (
                    pd.Timestamp(start_map[auth.loc[i, "card_id"]])
                    + pd.Timedelta(minutes=float(jit_min[j]))
                ).to_datetime64()
                for j, i in enumerate(cnp_idx)
            ],
            dtype="datetime64[ns]",
        )
        auth.loc[cnp_idx, "transmission_ts_utc"] = new_ts

    # --- 3.3 Counterfeit CP
    mask_cp = fraud_mask.copy()
    mask_cp[fraud_mask] = scenarios == "counterfeit_cp"

    auth.loc[mask_cp, "channel"] = "pos"
    auth.loc[mask_cp, "card_present"] = 1
    auth.loc[mask_cp, "pos_entry_mode"] = rng.choice(
        ["magstripe", "manual"],
        size=mask_cp.sum(),
        p=[0.80, 0.20],  # counterfeit casi siempre es magstripe clonado
    )
    auth.loc[mask_cp, "three_ds_status"] = "none"
    auth.loc[mask_cp, "eci"] = "none"
    auth.loc[mask_cp, "cvv_result"] = "U"
    auth.loc[mask_cp, "avs_result"] = "U"
    auth.loc[mask_cp, "device_id"] = "none"
    auth.loc[mask_cp, "ip_country"] = "none"
    auth.loc[mask_cp, "tokenized"] = 0
    auth.loc[mask_cp, "wallet_type"] = "none"

    cp_idx = np.where(mask_cp)[0]
    if len(cp_idx) > 0:
        hot = merchants[merchants["is_hot_merchant"] == 1].index.values
        chosen_hot = rng.choice(hot, size=len(cp_idx), replace=True)

        auth.loc[
            cp_idx,
            [
                "merchant_id",
                "terminal_id",
                "acquirer_id",
                "mcc",
                "merchant_country",
                "merchant_city",
                "merchant_name",
                "is_hot_merchant",
            ],
        ] = merchants.loc[
            chosen_hot,
            [
                "merchant_id",
                "terminal_id",
                "acquirer_id",
                "mcc",
                "merchant_country",
                "merchant_city",
                "merchant_name",
                "is_hot_merchant",
            ],
        ].to_numpy()

        abroad = rng.random(len(cp_idx)) < 0.25
        auth.loc[cp_idx[abroad], "merchant_country"] = rng.choice(
            ["AR", "PE", "CO", "BR", "US", "ES"],
            size=abroad.sum(),
            p=[0.20, 0.15, 0.15, 0.15, 0.20, 0.15],
        )

        burst_centers = sample_time_uniform(max(30, len(cp_idx) // 900))
        assigned = rng.integers(0, len(burst_centers), size=len(cp_idx))
        jit_min = rng.normal(loc=0, scale=18, size=len(cp_idx))
        new_ts = np.array(
            [
                (
                    pd.Timestamp(burst_centers[assigned[i]])
                    + pd.Timedelta(minutes=float(jit_min[i]))
                ).to_datetime64()
                for i in range(len(cp_idx))
            ],
            dtype="datetime64[ns]",
        )
        auth.loc[cp_idx, "transmission_ts_utc"] = new_ts

    # --- 3.4 Friendly fraud (transacción aparentemente legítima, luego disputada)
    mask_ff = fraud_mask.copy()
    mask_ff[fraud_mask] = scenarios == "friendly_fraud"
    ff_idx = np.where(mask_ff)[0]
    if len(ff_idx) > 0:
        to_ecom = rng.random(len(ff_idx)) < 0.65  # más ecommerce (refleja mix real)
        to_pos = ~to_ecom

        # Friendly fraud ecommerce: señales limpias (parece legítimo)
        auth.loc[ff_idx[to_ecom], "channel"] = "ecommerce"
        auth.loc[ff_idx[to_ecom], "card_present"] = 0
        auth.loc[ff_idx[to_ecom], "pos_entry_mode"] = "ecommerce"
        auth.loc[ff_idx[to_ecom], "three_ds_status"] = rng.choice(
            ["authenticated", "attempted"],
            size=to_ecom.sum(),
            p=[0.75, 0.25],
        )
        auth.loc[ff_idx[to_ecom], "cvv_result"] = rng.choice(
            ["M", "U"],
            size=to_ecom.sum(),
            p=[0.92, 0.08],
        )
        auth.loc[ff_idx[to_ecom], "eci"] = rng.choice(
            ["05", "06"],
            size=to_ecom.sum(),
            p=[0.75, 0.25],
        )

        # Friendly fraud POS: chip/contactless (no magstripe, es compra real)
        auth.loc[ff_idx[to_pos], "channel"] = "pos"
        auth.loc[ff_idx[to_pos], "card_present"] = 1
        auth.loc[ff_idx[to_pos], "pos_entry_mode"] = rng.choice(
            ["chip", "contactless"],
            size=to_pos.sum(),
            p=[0.58, 0.42],
        )
        auth.loc[ff_idx[to_pos], "three_ds_status"] = "none"
        auth.loc[ff_idx[to_pos], "eci"] = "none"
        auth.loc[ff_idx[to_pos], "cvv_result"] = rng.choice(
            ["M", "U"],
            size=to_pos.sum(),
            p=[0.85, 0.15],
        )
        auth.loc[ff_idx[to_pos], "device_id"] = "none"
        auth.loc[ff_idx[to_pos], "ip_country"] = "none"

    auth["dow"], auth["hour"], auth["month"] = dow_hour_month(auth["transmission_ts_utc"])
    ts_dt = pd.to_datetime(auth["transmission_ts_utc"])
    auth["local_txn_date"] = ts_dt.dt.date
    auth["local_txn_time"] = ts_dt.dt.time

    # =========================================================
    # 4) SIMULAR DECISIÓN DE AUTORIZACIÓN (response_code)
    # =========================================================
    risk = np.zeros(N_ROWS, dtype=float)

    risk += auth["pos_entry_mode"].isin(["manual", "magstripe"]).astype(float) * 0.45
    risk += auth["cvv_result"].isin(["N"]).astype(float) * 0.35
    risk += auth["three_ds_status"].isin(["failed"]).astype(float) * 0.40
    risk += (
        (auth["channel"] == "ecommerce")
        & (auth["ip_country"] != "none")
        & (auth["ip_country"] != "CL")
    ).astype(float) * 0.30
    risk += auth["is_hot_merchant"].astype(float) * 0.15
    risk += (auth["amount"] > 220000).astype(float) * 0.12

    risk += labels["fraud_attempt"].astype(float).values * 0.22

    risk += rng.normal(0, 0.05, size=N_ROWS)
    risk = np.clip(risk, 0, 1)

    auth["response_code"] = [response_from_risk(r) for r in risk]
    auth["auth_code"] = np.where(
        auth["response_code"].values == "00",
        [rand_id("A", 6) for _ in range(N_ROWS)],
        None,
    )

    fraud_declined = (labels["fraud_attempt"].values == 1) & (
        auth["response_code"].values != "00"
    )
    flip_to_approve = fraud_declined & (rng.random(N_ROWS) < 0.22)
    auth.loc[flip_to_approve, "response_code"] = "00"
    auth.loc[flip_to_approve, "auth_code"] = [
        rand_id("A", 6) for _ in range(flip_to_approve.sum())
    ]

    good_declined = (labels["fraud_attempt"].values == 0) & (
        auth["response_code"].values != "00"
    )
    flip_to_approve_good = good_declined & (rng.random(N_ROWS) < 0.14)
    auth.loc[flip_to_approve_good, "response_code"] = "00"
    auth.loc[flip_to_approve_good, "auth_code"] = [
        rand_id("A", 6) for _ in range(flip_to_approve_good.sum())
    ]

    auth.sort_values("transmission_ts_utc", inplace=True)
    labels = labels.set_index("txn_id").loc[auth["txn_id"].values].reset_index()

    # =========================================================
    # 5) FEATURES DE VELOCIDAD (SIN LEAKAGE)
    # =========================================================
    def build_velocity_features(auth_df: pd.DataFrame) -> pd.DataFrame:
        df = auth_df[
            [
                "txn_id",
                "transmission_ts_utc",
                "card_id",
                "device_id",
                "merchant_id",
                "amount",
                "channel",
                "merchant_country",
                "response_code",
            ]
        ].copy()
        # Codificar merchant_id a un código numérico para poder usar rolling.apply
        df["merchant_id_code"] = df["merchant_id"].astype("category").cat.codes
        df["transmission_ts_utc"] = pd.to_datetime(df["transmission_ts_utc"])
        df.sort_values("transmission_ts_utc", inplace=True)

        def per_group(g, prefix):
            g = g.sort_values("transmission_ts_utc").copy()
            g = g.set_index("transmission_ts_utc")

            g[f"{prefix}_txn_count_1h"] = g["txn_id"].rolling("1h").count().shift(1)
            g[f"{prefix}_txn_count_24h"] = g["txn_id"].rolling("24h").count().shift(1)
            g[f"{prefix}_sum_amount_24h"] = g["amount"].rolling("24h").sum().shift(1)

            # Usar merchant_id_code numérico para evitar errores de pandas 3.0
            g[f"{prefix}_unique_merchants_24h"] = (
                g["merchant_id_code"]
                .rolling("24h")
                .apply(lambda x: len(pd.unique(x)), raw=False)
                .shift(1)
            )

            g[f"{prefix}_time_since_prev_min"] = (
                g.index.to_series().diff().dt.total_seconds().div(60)
            )
            g[f"{prefix}_time_since_prev_min"] = g[
                f"{prefix}_time_since_prev_min"
            ].fillna(999999)

            g["_decl"] = (g["response_code"] != "00").astype(int)
            g[f"{prefix}_pct_declines_prev30"] = (
                g["_decl"].rolling(30).mean().shift(1)
            )

            out = g.reset_index()[
                [
                    "txn_id",
                    f"{prefix}_txn_count_1h",
                    f"{prefix}_txn_count_24h",
                    f"{prefix}_sum_amount_24h",
                    f"{prefix}_unique_merchants_24h",
                    f"{prefix}_time_since_prev_min",
                    f"{prefix}_pct_declines_prev30",
                ]
            ]
            return out

        card_feat = df.groupby("card_id", group_keys=False).apply(
            lambda g: per_group(g, "card")
        )

        df_dev = df[df["device_id"] != "none"].copy()
        if len(df_dev) > 0:
            dev_feat = df_dev.groupby("device_id", group_keys=False).apply(
                lambda g: per_group(g, "device")
            )
        else:
            dev_feat = pd.DataFrame({"txn_id": []})

        feat = pd.DataFrame({"txn_id": df["txn_id"].values})
        feat = feat.merge(card_feat, on="txn_id", how="left")
        feat = feat.merge(dev_feat, on="txn_id", how="left")

        feat["card_new_merchant_24h_flag"] = (
            feat["card_unique_merchants_24h"].fillna(0) == 0
        ).astype(int)
        if "device_unique_merchants_24h" in feat.columns:
            feat["device_new_merchant_24h_flag"] = (
                feat["device_unique_merchants_24h"].fillna(0) == 0
            ).astype(int)
        else:
            feat["device_new_merchant_24h_flag"] = 1

        for c in feat.columns:
            if c.endswith(
                (
                    "count_1h",
                    "count_24h",
                    "sum_amount_24h",
                    "unique_merchants_24h",
                    "pct_declines_prev30",
                )
            ):
                feat[c] = feat[c].fillna(0.0)
            if c.endswith("time_since_prev_min"):
                feat[c] = feat[c].fillna(999999.0)

        return feat

    features = build_velocity_features(auth)

    auth_events = auth.copy()
    auth_events.to_csv(RAW_DATA_DIR / "auth_events.csv", index=False)

    labels.to_csv(RAW_DATA_DIR / "labels.csv", index=False)

    ml = (
        auth_events.merge(features, on="txn_id", how="left").merge(
            labels[
                [
                    "txn_id",
                    "fraud_attempt",
                    "fraud_scenario",
                    "label_source",
                    "days_to_confirm",
                ]
            ],
            on="txn_id",
            how="left",
        )
    )
    ml.to_csv(RAW_DATA_DIR / "ml_dataset.csv", index=False)

    dataset_profile = {
        "issuer_profile": {
            "issuer_id": ISSUER_ID,
            "issuer_name": ISSUER_NAME,
            "issuer_country": HOME_COUNTRY,
            "issuer_id_nunique": int(ml["issuer_id"].nunique()),
            "home_country_nunique": int(ml["home_country"].nunique()),
        },
        "dataset_shape": {
            "rows": int(len(ml)),
            "columns": int(ml.shape[1]),
        },
        "date_range_utc": {
            "start": str(pd.to_datetime(ml["transmission_ts_utc"]).min()),
            "end": str(pd.to_datetime(ml["transmission_ts_utc"]).max()),
        },
        "channel_mix": ml["channel"].value_counts(normalize=True).round(4).to_dict(),
        "fraud": {
            "fraud_rate": float(ml["fraud_attempt"].mean()),
            "fraud_scenarios": (
                ml.loc[ml["fraud_attempt"] == 1, "fraud_scenario"]
                .value_counts(normalize=True)
                .round(4)
                .to_dict()
            ),
        },
        "approved_rate": float((ml["response_code"] == "00").mean()),
    }
    with (RAW_DATA_DIR / "dataset_profile.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_profile, f, indent=2)

    print("OK files: auth_events.csv, labels.csv, ml_dataset.csv, dataset_profile.json")
    print("Rows:", len(ml))
    print("Fraud rate:", ml["fraud_attempt"].mean())
    print("Approved rate:", (ml["response_code"] == "00").mean())
    print(
        "Fraud approved rate:",
        ((ml["fraud_attempt"] == 1) & (ml["response_code"] == "00")).mean(),
    )
    print(
        "Fraud declined rate:",
        ((ml["fraud_attempt"] == 1) & (ml["response_code"] != "00")).mean(),
    )
    print(
        "Top fraud scenarios:\n",
        ml.loc[ml["fraud_attempt"] == 1, "fraud_scenario"]
        .value_counts(normalize=True)
        .head(),
    )


if __name__ == "__main__":
    main()

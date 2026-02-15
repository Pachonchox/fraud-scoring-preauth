# Fraud Scoring Pre-autorización — Emisor Mastercard Chile

Sistema de machine learning para detectar fraude **antes** de que la transacción sea autorizada, usando únicamente información disponible en el mensaje ISO 8583.

**[Ver demo en vivo](https://fraud-scoring-preauth.streamlit.app)**

---

## Resultados

| Métrica | Valor |
|---------|-------|
| Modelo ganador | HistGradientBoosting |
| ROC AUC | 0.958 |
| PR AUC | 0.797 |
| F1 fraude | 0.797 |
| Precisión fraude | 90.2% |
| Recall fraude | 71.3% |
| Threshold operativo | 0.95 |

Comparación de los 3 modelos entrenados:

| Modelo | ROC AUC | PR AUC | F1 | Precisión | Recall |
|--------|---------|--------|----|-----------|--------|
| **HistGradientBoosting** | **0.9582** | **0.7966** | **0.7967** | **0.9024** | **0.7131** |
| RandomForest | 0.9558 | 0.7843 | 0.7648 | 0.8757 | 0.6788 |
| MLPClassifier | 0.9563 | 0.7398 | 0.6693 | 0.6132 | 0.7366 |

## El problema

Cada vez que un tarjetahabiente pasa su tarjeta o compra en línea, el emisor tiene milisegundos para decidir si aprueba o rechaza. Este proyecto construye un modelo de scoring que evalúa la probabilidad de fraude usando solo la información del mensaje de autorización, *antes* de enviar la respuesta.

## Dataset

- **300,000 transacciones** sintéticas con distribuciones realistas (estándares Mastercard 2025)
- **Emisor único chileno** — simula el caso real de un banco
- **Ecommerce 58% / POS 42%** — sin ATM (caso de uso: compras)
- **Tasa de fraude: 0.8%** — realista para intentos en LATAM
- **4 patrones de fraude**: card testing, CNP takeover, counterfeit CP, friendly fraud

### Entry modes coherentes por canal

| Canal | Entry Mode | Proporción | Justificación |
|-------|-----------|------------|---------------|
| POS | chip | 55% | EMV dominante en Chile |
| POS | contactless | 38% | Alta adopción NFC |
| POS | magstripe | 5% | Terminales legacy |
| POS | manual | 2% | MOTO residual |
| Ecommerce | ecommerce | 72% | Compra web estándar |
| Ecommerce | credential_on_file | 18% | Suscripciones, COF |
| Ecommerce | in_app | 10% | App móvil |

## Feature engineering (anti-leakage)

Se construyen **38+ features** exclusivamente con información pre-autorización:

| Tipo | Ejemplos |
|------|----------|
| Transaccionales | amount, log_amount, channel, mcc |
| Autenticación | cvv_result, three_ds_status, eci, tokenized |
| Geográficas | merchant_country, ip_country, foreign flags |
| Temporales | dow, hour, month, night_txn_flag |
| Velocidad tarjeta | card_txn_count_1h/24h, card_sum_amount_24h |
| Velocidad dispositivo | device_txn_count_1h/24h, device_sum_amount_24h |
| Flags de riesgo | risky_entry_mode, cvv_negative, card_decline_spike |

**Principio clave**: ningún feature usa información post-decisión (response_code, auth_code).

## Stack tecnológico

| Componente | Tecnología |
|------------|-----------|
| Lenguaje | Python 3.11+ |
| ML | scikit-learn (HGB, RF, MLP) |
| Datos | pandas + pyarrow (Parquet) |
| Dashboard | Streamlit + Altair |
| Ejecución | Apple M2 Pro (~5 min pipeline completo) |

## Estructura del proyecto

```
├── app_streamlit.py              # Dashboard interactivo (4 tabs)
├── src/
│   ├── config.py                 # Configuración central
│   ├── data/
│   │   └── generate_dataset.py   # Generación de datos sintéticos
│   ├── features/
│   │   └── build_features.py     # Feature engineering pre-autorización
│   └── models/
│       └── train_pipeline.py     # Entrenamiento RF + HGB + MLP
├── data/
│   ├── raw/                      # dataset_profile.json (CSVs se regeneran)
│   └── processed/                # ml_dataset_preauth.parquet
├── models/                       # Modelos serializados + métricas
└── .streamlit/
    └── config.toml               # Tema del dashboard
```

## Reproducir el proyecto

```bash
# Clonar
git clone https://github.com/Pachonchox/fraud-scoring-preauth.git
cd fraud-scoring-preauth

# Entorno
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pipeline completo (~5 min en M2 Pro)
python -m src.data.generate_dataset      # Genera 300K transacciones
python -m src.features.build_features    # Construye features
python -m src.models.train_pipeline      # Entrena 3 modelos

# Dashboard
streamlit run app_streamlit.py
```

## Dashboard

El dashboard tiene 4 tabs narrativos:

1. **El problema** — contexto, dataset, patrones de fraude, decisiones de diseño
2. **El modelo** — feature engineering, comparación de modelos, hiperparámetros, stack
3. **Impacto** — simulador de costos, curva precision/recall, casos de uso
4. **Demo** — scoring en vivo sobre 6 escenarios predefinidos

---

Desarrollado como proyecto de portfolio en ciencia de datos aplicada a la industria de pagos.

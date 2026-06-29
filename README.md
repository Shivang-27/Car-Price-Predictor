# 🚗 Car Price Predictor

A Random Forest regression model that predicts used car prices based on manufacturer, model, fuel type, engine size, year of manufacture, and mileage — trained on 50,000 real car listings.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter)](https://jupyter.org)

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | `car_sales_data.csv` |
| **Size** | 50,000 car listings |
| **Target** | Price (£) |
| **Price Range** | £76 – £168,081 |
| **Median Price** | £7,972 |

### Correlation with Price (before encoding)

| Feature | Correlation |
|---|---|
| Year of Manufacture | +0.714 |
| Engine Size | +0.398 |
| Mileage | -0.633 |

> Newer cars and larger engines push prices up; higher mileage pulls them down — all intuitive and confirmed by the data.

---

## Features

| Feature | Type | Description |
|---|---|---|
| `Manufacturer` | Categorical | BMW, Ford, Porsche, Toyota, VW |
| `Model` | Categorical | Specific model (e.g. Golf, M5, Prius) |
| `Fuel type` | Categorical | Petrol / Diesel / Hybrid |
| `Engine size` | Numerical | Engine displacement in litres |
| `Year of manufacture` | Numerical | Production year (1984–2022) |
| `Mileage` | Numerical | Odometer reading |

---

## Pipeline

A `ColumnTransformer` pipeline was used to cleanly preprocess both numerical and categorical features before training:

- **Numerical** — Median imputation + Standard scaling
- **Categorical** — One-Hot Encoding

Stratified split on year-of-manufacture bands ensured balanced train/test distribution across car eras.

---

## Model

```
Algorithm  : Random Forest Regressor (default hyperparameters)
Train/Test : 80% / 20%  (stratified by year of manufacture)
```

---

## Results

| Metric | Value |
|---|---|
| **R² (Test)** | **0.998** |
| **MAE** | £292 |
| **RMSE** | £658 |
| Train R² | 0.9998 |

> An R² of 0.998 means the model explains 99.8% of the variance in used car prices. The tight gap between train and test R² confirms the model is not significantly overfitting.

---

## Feature Importance (Top 15)

| Feature | Importance |
|---|---|
| `year of manufacture` | 0.6194 |
| `engine size` | 0.2590 |
| `mileage` | 0.0397 |
| `Model_M5` | 0.0219 |
| `Model_911` | 0.0216 |
| `Fuel type_Petrol` | 0.0095 |
| `Model_RAV4` | 0.0084 |
| `Manufacturer_BMW` | 0.0025 |
| `Manufacturer_Porsche` | 0.0024 |
| `Model_Polo` | 0.0019 |
| `Manufacturer_Toyota` | 0.0018 |
| `Model_Golf` | 0.0016 |
| `Model_Passat` | 0.0015 |
| `Model_Prius` | 0.0014 |
| `Model_Focus` | 0.0014 |

> Year of manufacture alone accounts for ~62% of the model's predictive power, followed by engine size at ~26%. This makes intuitive sense — car age is the single biggest driver of depreciation. Prestige models (M5, 911) show up as meaningful signals despite being rare in the data.

---

## 🖥️ Streamlit Web App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shivang-27-car-price-predictor-car-price-predictor-app-jczswg.streamlit.app/)

The trained model is served as an interactive web app built with Streamlit (`car_price_predictor_app.py`). The app retrains the full pipeline at startup from the raw CSV — no separate `.pkl` file needed.

### Features

**Predict Price tab** — Select manufacturer, model, fuel type, year, engine size, and mileage. Models filter dynamically by manufacturer. Returns an estimated price with a ±MAE confidence range.

**Model Performance tab** — Test-set metrics (R², MAE, RMSE), actual vs. predicted scatter plot, residuals distribution histogram, and per-price-bucket error breakdown (<£5K, £5K–15K, £15K–40K, >£40K). Includes a train vs. test R² overfitting check.

**Feature Analysis tab** — Top 15 feature importances bar chart, key driver insight cards, and a numeric correlation heatmap (Engine Size, Year, Mileage vs. Price).

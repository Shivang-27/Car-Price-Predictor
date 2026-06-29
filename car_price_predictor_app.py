import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark background */
  .stApp { background-color: #0d0f14; color: #e8eaf0; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #13151d !important;
    border-right: 1px solid #1e2130;
  }
  [data-testid="stSidebar"] * { color: #c8cad4 !important; }

  /* Remove default padding */
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }

  /* Metric cards */
  .metric-card {
    background: #13151d;
    border: 1px solid #1e2130;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
  }
  .metric-card .label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7280;
    margin-bottom: 0.4rem;
  }
  .metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #4ade80;
    line-height: 1;
  }
  .metric-card .sub {
    font-size: 0.72rem;
    color: #6b7280;
    margin-top: 0.35rem;
  }

  /* Price result box */
  .price-box {
    background: linear-gradient(135deg, #0f1f14 0%, #0d1a24 100%);
    border: 1px solid #22c55e44;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
  }
  .price-box .price-label { font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase; color: #6b7280; }
  .price-box .price-value { font-family: 'Space Mono', monospace; font-size: 3.2rem; font-weight: 700; color: #4ade80; margin: 0.3rem 0; }
  .price-box .price-range { font-size: 0.82rem; color: #9ca3af; }

  /* Section header */
  .section-header {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2130;
  }

  /* Hero strip */
  .hero {
    background: #13151d;
    border: 1px solid #1e2130;
    border-radius: 16px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .hero h1 { font-size: 1.9rem; font-weight: 700; color: #f1f3f9; margin: 0; }
  .hero p  { font-size: 0.9rem; color: #6b7280; margin: 0.3rem 0 0; }
  .hero .badge {
    background: #0f2119;
    border: 1px solid #22c55e55;
    border-radius: 100px;
    padding: 0.35rem 1rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #4ade80;
    letter-spacing: 0.05em;
  }

  /* Stacked info row */
  .info-row {
    background: #13151d;
    border: 1px solid #1e2130;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.7rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .info-row .key { font-size: 0.82rem; color: #9ca3af; }
  .info-row .val { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #e8eaf0; font-weight: 600; }

  /* Tab styling */
  [data-baseweb="tab-list"] { background: transparent !important; gap: 0.5rem; }
  [data-baseweb="tab"] {
    background: #13151d !important;
    border: 1px solid #1e2130 !important;
    border-radius: 8px !important;
    color: #6b7280 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1rem !important;
  }
  [aria-selected="true"] {
    background: #0f2119 !important;
    border-color: #22c55e55 !important;
    color: #4ade80 !important;
  }

  /* Selectbox / inputs */
  [data-testid="stSelectbox"] > div > div { background: #13151d !important; border-color: #1e2130 !important; }
  [data-testid="stNumberInput"] > div > div > input { background: #13151d !important; border-color: #1e2130 !important; color: #e8eaf0 !important; }
  .stSlider [data-testid="stThumbValue"] { color: #4ade80 !important; }

  /* Button */
  .stButton > button {
    background: #16a34a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 2rem !important;
    width: 100%;
    transition: background 0.2s;
  }
  .stButton > button:hover { background: #15803d !important; }

  /* Matplotlib transparent */
  .element-container { background: transparent !important; }

  h2, h3 { color: #e8eaf0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Data & model (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on 50 000 cars…")
def load_model_and_data():
    cars = pd.read_csv("data_files/car_sales_data.csv")

    # Stratified split
    cars["YOM_cat"] = pd.cut(
        cars["Year of manufacture"],
        bins=[1980, 1996, 2004, 2012, 2022],
        labels=[1, 2, 3, 4],
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for tr_idx, te_idx in split.split(cars, cars["YOM_cat"]):
        train = cars.loc[tr_idx].drop("YOM_cat", axis=1)
        test  = cars.loc[te_idx].drop("YOM_cat", axis=1)

    num_attribs = ["Engine size", "Year of manufacture", "Mileage"]
    cat_attribs = ["Manufacturer", "Model", "Fuel type"]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    full_pipeline = ColumnTransformer([
        ("num", num_pipe, num_attribs),
        ("cat", cat_pipe, cat_attribs),
    ])

    X_train = train.drop("Price", axis=1)
    y_train = train["Price"]
    X_test  = test.drop("Price", axis=1)
    y_test  = test["Price"]

    X_train_prep = full_pipeline.fit_transform(X_train)
    X_test_prep  = full_pipeline.transform(X_test)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_prep, y_train)

    train_preds = model.predict(X_train_prep)
    test_preds  = model.predict(X_test_prep)

    metrics = {
        "mae":       mean_absolute_error(y_test, test_preds),
        "mse":       mean_squared_error(y_test, test_preds),
        "rmse":      np.sqrt(mean_squared_error(y_test, test_preds)),
        "r2":        r2_score(y_test, test_preds),
        "train_r2":  r2_score(y_train, train_preds),
        "train_rmse": root_mean_squared_error(y_train, train_preds),
    }

    feature_names = full_pipeline.get_feature_names_out()
    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    return model, full_pipeline, cars, y_test, test_preds, metrics, importance


model, pipeline, cars, y_test, test_preds, metrics, importance = load_model_and_data()

# Dropdown options from data
manufacturers = sorted(cars["Manufacturer"].dropna().unique().tolist())
models_all    = sorted(cars["Model"].dropna().unique().tolist())
fuel_types    = sorted(cars["Fuel type"].dropna().unique().tolist())
year_min, year_max = int(cars["Year of manufacture"].min()), int(cars["Year of manufacture"].max())
mile_min, mile_max = int(cars["Mileage"].min()), int(cars["Mileage"].max())
eng_min,  eng_max  = float(cars["Engine size"].min()), float(cars["Engine size"].max())


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div>
    <h1>🚗 Car Price Predictor</h1>
    <p>Random Forest · 50 000 cars · Instant valuation</p>
  </div>
  <div class="badge">R² = 0.998</div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮  Predict Price", "📊  Model Performance", "📈  Feature Analysis"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-header">Car Details</div>', unsafe_allow_html=True)

        manufacturer = st.selectbox("Manufacturer", manufacturers, index=manufacturers.index("Ford") if "Ford" in manufacturers else 0)
        filtered_models = sorted(cars[cars["Manufacturer"] == manufacturer]["Model"].dropna().unique().tolist())
        car_model = st.selectbox("Model", filtered_models)
        fuel_type = st.selectbox("Fuel Type", fuel_types)

        col_a, col_b = st.columns(2)
        with col_a:
            year = st.number_input("Year of Manufacture", min_value=year_min, max_value=year_max, value=2015, step=1)
        with col_b:
            engine_size = st.number_input("Engine Size (L)", min_value=eng_min, max_value=eng_max, value=1.6, step=0.1, format="%.1f")

        mileage = st.slider("Mileage (km)", min_value=mile_min, max_value=mile_max, value=80000, step=1000)

        st.markdown("")
        predict_btn = st.button("Estimate Price →")

    with right:
        st.markdown('<div class="section-header">Valuation Result</div>', unsafe_allow_html=True)

        if predict_btn:
            input_df = pd.DataFrame([{
                "Manufacturer":         manufacturer,
                "Model":                car_model,
                "Engine size":          engine_size,
                "Fuel type":            fuel_type,
                "Year of manufacture":  year,
                "Mileage":              mileage,
            }])
            prepared = pipeline.transform(input_df)
            pred     = model.predict(prepared)[0]

            # Confidence-style range (±MAE)
            lo, hi = max(0, pred - metrics["mae"]), pred + metrics["mae"]

            st.markdown(f"""
            <div class="price-box">
              <div class="price-label">Estimated Market Value</div>
              <div class="price-value">£{pred:,.0f}</div>
              <div class="price-range">Typical range: £{lo:,.0f} – £{hi:,.0f} &nbsp;(±MAE £{metrics['mae']:,.0f})</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Input Summary</div>', unsafe_allow_html=True)
            for k, v in [("Manufacturer", manufacturer), ("Model", car_model), ("Fuel", fuel_type),
                         ("Year", year), ("Engine", f"{engine_size:.1f} L"), ("Mileage", f"{mileage:,} km")]:
                st.markdown(f'<div class="info-row"><span class="key">{k}</span><span class="val">{v}</span></div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="price-box" style="opacity:0.45; padding: 3rem 2rem;">
              <div class="price-label">Awaiting Input</div>
              <div class="price-value" style="font-size:1.5rem; color:#374151;">Fill in the details<br>and click Estimate</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Test-Set Metrics (10 000 held-out cars)</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "R² Score",      f"{metrics['r2']:.4f}",         "Near-perfect fit"),
        (c2, "MAE",           f"£{metrics['mae']:,.0f}",       "Avg. absolute error"),
        (c3, "RMSE",          f"£{metrics['rmse']:,.0f}",      "Root mean sq. error"),
        (c4, "Train R²",      f"{metrics['train_r2']:.4f}",    "Overfitting check"),
    ]
    for col, label, val, sub in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value">{val}</div>
              <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Actual vs Predicted scatter ───────────────────────────────────────────
    left2, right2 = st.columns(2, gap="large")

    with left2:
        st.markdown('<div class="section-header">Actual vs Predicted Prices</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#13151d")
        ax.set_facecolor("#0d0f14")

        sample_n = min(3000, len(y_test))
        idx = np.random.choice(len(y_test), sample_n, replace=False)
        y_s, p_s = np.array(y_test)[idx], np.array(test_preds)[idx]

        ax.scatter(y_s, p_s, alpha=0.35, s=8, color="#4ade80", edgecolors="none")
        lims = [min(y_test.min(), min(test_preds)), max(y_test.max(), max(test_preds))]
        ax.plot(lims, lims, "r--", linewidth=1.5, alpha=0.7, label="Perfect fit")

        ax.set_xlabel("Actual Price (£)", color="#9ca3af", fontsize=9)
        ax.set_ylabel("Predicted Price (£)", color="#9ca3af", fontsize=9)
        ax.tick_params(colors="#6b7280", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2130")
        ax.legend(framealpha=0, labelcolor="#9ca3af", fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right2:
        st.markdown('<div class="section-header">Residuals Distribution</div>', unsafe_allow_html=True)

        residuals = np.array(y_test) - np.array(test_preds)

        fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor="#13151d")
        ax2.set_facecolor("#0d0f14")
        ax2.hist(residuals, bins=80, color="#4ade80", alpha=0.7, edgecolor="none")
        ax2.axvline(0, color="#ef4444", linewidth=1.5, linestyle="--", alpha=0.8)
        ax2.set_xlabel("Residual (Actual − Predicted, £)", color="#9ca3af", fontsize=9)
        ax2.set_ylabel("Count", color="#9ca3af", fontsize=9)
        ax2.tick_params(colors="#6b7280", labelsize=8)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#1e2130")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── Error breakdown ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Error Breakdown by Price Bucket</div>', unsafe_allow_html=True)

    df_err = pd.DataFrame({"actual": y_test, "pred": test_preds})
    df_err["bucket"] = pd.cut(df_err["actual"], bins=[0, 5000, 15000, 40000, 200000],
                              labels=["<£5K", "£5K–15K", "£15K–40K", ">£40K"])
    bucket_stats = df_err.groupby("bucket", observed=True).apply(
        lambda g: pd.Series({
            "MAE":  mean_absolute_error(g["actual"], g["pred"]),
            "RMSE": np.sqrt(mean_squared_error(g["actual"], g["pred"])),
            "R²":   r2_score(g["actual"], g["pred"]),
            "N":    len(g),
        })
    ).reset_index()

    # Display as styled metric rows
    cols_b = st.columns(4)
    for i, row in bucket_stats.iterrows():
        with cols_b[i]:
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">{row['bucket']}</div>
              <div class="value" style="font-size:1.3rem;">R²&nbsp;{row['R²']:.3f}</div>
              <div class="sub">MAE £{row['MAE']:,.0f} · n={int(row['N']):,}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Overfitting summary ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Overfitting Check</div>', unsafe_allow_html=True)
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Train R²</div>
          <div class="value">{metrics['train_r2']:.4f}</div>
          <div class="sub">40 000 training cars</div>
        </div>""", unsafe_allow_html=True)
    with oc2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Test R²</div>
          <div class="value">{metrics['r2']:.4f}</div>
          <div class="sub">10 000 held-out cars</div>
        </div>""", unsafe_allow_html=True)
    with oc3:
        gap = metrics['train_r2'] - metrics['r2']
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">Generalisation Gap</div>
          <div class="value" style="font-size:1.5rem;">{gap:.4f}</div>
          <div class="sub">{"✅ Minimal overfit" if gap < 0.005 else "⚠️ Some overfit"}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Top 15 Feature Importances</div>', unsafe_allow_html=True)

    top15 = importance.head(15).sort_values()

    # Clean display names
    def clean_name(n):
        n = n.replace("num__", "").replace("cat__", "")
        return n.replace("_", " ").title()

    labels = [clean_name(n) for n in top15.index]
    vals   = top15.values
    colors = ["#4ade80" if v > 0.05 else "#22c55e" if v > 0.01 else "#166534" for v in vals]

    fig3, ax3 = plt.subplots(figsize=(9, 5), facecolor="#13151d")
    ax3.set_facecolor("#0d0f14")
    bars = ax3.barh(labels, vals, color=colors, height=0.65)
    ax3.set_xlabel("Importance Score", color="#9ca3af", fontsize=9)
    ax3.tick_params(axis="y", colors="#c8cad4", labelsize=8.5)
    ax3.tick_params(axis="x", colors="#6b7280", labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#1e2130")

    # Value labels
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", ha="left", color="#9ca3af", fontsize=7.5)

    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Insight cards ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)

    top3 = importance.head(3)
    pct  = (top3.values / top3.values.sum() * 100)

    i1, i2, i3 = st.columns(3)
    for col, (feat, imp), p in zip([i1, i2, i3], zip(top3.index, top3.values), pct):
        with col:
            label = clean_name(feat)
            st.markdown(f"""
            <div class="metric-card">
              <div class="label">#{list(top3.index).index(feat)+1} Driver</div>
              <div class="value" style="font-size:1.4rem;">{imp:.3f}</div>
              <div class="sub">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Correlation heatmap (numeric) ─────────────────────────────────────────
    st.markdown('<div class="section-header">Numeric Feature Correlations with Price</div>', unsafe_allow_html=True)

    corr = cars[["Engine size", "Year of manufacture", "Mileage", "Price"]].corr()
    fig4, ax4 = plt.subplots(figsize=(6, 4), facecolor="#13151d")
    ax4.set_facecolor("#0d0f14")

    im = ax4.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax4.set_xticks(range(len(corr.columns)))
    ax4.set_yticks(range(len(corr.columns)))
    ax4.set_xticklabels(corr.columns, color="#9ca3af", fontsize=8, rotation=30, ha="right")
    ax4.set_yticklabels(corr.columns, color="#9ca3af", fontsize=8)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#1e2130")

    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax4.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                     color="#0d0f14" if abs(corr.values[i,j]) > 0.4 else "#000000", fontsize=8.5, fontweight="bold")

    plt.colorbar(im, ax=ax4, fraction=0.035, pad=0.04).ax.tick_params(colors="#6b7280", labelsize=7)
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:#1e2130; margin-top:2.5rem; margin-bottom:1rem;">
<p style="text-align:center; color:#374151; font-size:0.75rem; letter-spacing:0.08em;">
  RANDOM FOREST REGRESSOR &nbsp;·&nbsp; SKLEARN &nbsp;·&nbsp; 50 000 SAMPLES &nbsp;·&nbsp; 80/20 STRATIFIED SPLIT
</p>
""", unsafe_allow_html=True)

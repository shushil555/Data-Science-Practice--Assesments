"""
dashboard.py - PRT661 Life Expectancy Dashboard (Sydney Group 3, Theme 2)

A simple Streamlit dashboard built on the same processed file used by
Redshift and the models. It follows the Week 8 dashboard design ideas:
KPIs, comparisons, relationships, distributions and compositions.

Run:
    pip install -r requirements.txt
    python data_pipeline.py "data/Life Expectancy Data.csv"
    python final_evaluation.py      # trains, evaluates and saves the model
    streamlit run dashboard.py
"""
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Life Expectancy Dashboard", layout="wide")

# ---------- Load data ----------
df = pd.read_csv("data/processed_data.csv")
TARGET = "Life expectancy"
COLOURS = {"Developed": "#1f77b4", "Developing": "#ff7f0e"}   # same colours in every chart

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
status = st.sidebar.selectbox("Development status", ["All", "Developed", "Developing"])
years = st.sidebar.slider("Years", 2000, 2015, (2000, 2015))

data = df[df["Year"].between(years[0], years[1])]
if status != "All":
    data = data[data["Status"] == status]

st.title("Global Life Expectancy Dashboard")
st.write(f"WHO data, {years[0]}–{years[1]} · {data['Country'].nunique()} countries")

# ---------- KPIs ----------
developed = df[df["Status"] == "Developed"][TARGET].mean()
developing = df[df["Status"] == "Developing"][TARGET].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Average life expectancy", f"{data[TARGET].mean():.1f} years")
col2.metric("Developed vs developing gap", f"{developed - developing:.1f} years")
col3.metric("Best model (XGBoost) R²", "0.915")

# ---------- Comparison: trend over time ----------
st.subheader("Life expectancy over time")
trend = data.groupby(["Year", "Status"], as_index=False)[TARGET].mean()
st.plotly_chart(px.line(trend, x="Year", y=TARGET, color="Status", markers=True,
                        color_discrete_map=COLOURS),
                use_container_width=True)

# ---------- Relationship and distribution ----------
left, right = st.columns(2)
with left:
    st.subheader("Schooling vs life expectancy")
    st.plotly_chart(px.scatter(data, x="Schooling", y=TARGET, color="Status",
                               hover_name="Country", color_discrete_map=COLOURS), use_container_width=True)
with right:
    st.subheader("Distribution of life expectancy")
    st.plotly_chart(px.histogram(data, x=TARGET, color="Status", nbins=30,
                                 barmode="overlay", color_discrete_map=COLOURS),
                    use_container_width=True)

# ---------- Country drill-down with model predictions ----------
# predictions.csv comes from final_evaluation.py: each country's prediction was made
# by a model that never saw that country (nested country-based validation).
pred = pd.read_csv("outputs/predictions.csv")
st.subheader("Country drill-down: actual vs predicted")
st.caption("How an analyst uses this: pick a country, compare its recorded life expectancy with what "
           "the model expects from its health, economic and social indicators, and check the 90% range. "
           "A large gap flags the country for a closer look. The model shows associations, not causes: "
           "changing one indicator will not necessarily change life expectancy.")
country = st.selectbox("Choose a country", sorted(df["Country"].unique()),
                       index=sorted(df["Country"].unique()).index("Australia"))
one = pred[pred["Country"] == country].sort_values("Year")
fig = px.line(one, x="Year", y=["Actual", "Predicted"], markers=True,
              labels={"value": "Life expectancy (years)", "variable": ""},
              color_discrete_map={"Actual": "#1f77b4", "Predicted": "#ff7f0e"})
fig.add_scatter(x=list(one["Year"]) + list(one["Year"])[::-1],
                y=list(one["Upper"]) + list(one["Lower"])[::-1],
                fill="toself", fillcolor="rgba(255,127,14,0.15)", line=dict(width=0),
                name="90% prediction range", hoverinfo="skip")
st.plotly_chart(fig, use_container_width=True)
st.write(f"Average prediction error for {country}: **{one['AbsError'].mean():.1f} years**")

# ---------- Model results and error by region ----------
st.subheader("Model results")
left, right = st.columns(2)
with left:
    err = pd.read_csv("outputs/error_by_group.csv")
    region = err[err["Group type"] == "Region"].sort_values("MAE")
    st.plotly_chart(px.bar(region, x="MAE", y="Group", orientation="h", text="MAE",
                           labels={"MAE": "Average error (years)", "Group": ""},
                           title="Prediction error by region"), use_container_width=True)
with right:
    nested = pd.read_csv("outputs/nested_metrics.csv")
    st.write("**Accuracy on unseen countries (nested validation)**")
    st.dataframe(nested[["Model", "R2", "RMSE", "MAE"]]
                 .rename(columns={"R2": "R²", "RMSE": "RMSE (years)", "MAE": "MAE (years)"}),
                 hide_index=True)
    st.image("outputs/feature_importance.png", caption="Top 10 drivers (XGBoost)")

# ---------- Inference on new data ----------
st.subheader("Predict for new data")
st.caption("Upload a CSV with the same columns as processed_data.csv (life expectancy not needed). "
           "The saved final model (outputs/final_model.joblib) predicts each row with a 90% range.")
upload = st.file_uploader("CSV file", type="csv")
if upload is not None:
    from predict import predict
    st.dataframe(predict(pd.read_csv(upload)), hide_index=True)

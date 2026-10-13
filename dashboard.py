"""
dashboard.py - PRT661 Life Expectancy Dashboard (Sydney Group 3, Theme 2)

A simple Streamlit dashboard built on the same processed file used by
Redshift and the models. It follows the Week 8 dashboard design ideas:
KPIs, comparisons, relationships, distributions and compositions.

Run:
    pip install streamlit plotly pandas
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

# ---------- Country drill-down ----------
st.subheader("Country drill-down")
country = st.selectbox("Choose a country", sorted(df["Country"].unique()),
                       index=sorted(df["Country"].unique()).index("Australia"))
one = df[df["Country"] == country]
st.plotly_chart(px.line(one, x="Year", y=TARGET, markers=True,
                        title=f"{country}: life expectancy 2000–2015"),
                use_container_width=True)

# ---------- Composition: model results ----------
st.subheader("Model results")
left, right = st.columns(2)
with left:
    st.image("outputs/feature_importance.png", caption="Top 10 drivers (XGBoost)")
with right:
    metrics = pd.read_csv("outputs/model_metrics.csv")
    st.dataframe(metrics[["Model", "GroupCV_R2", "GroupCV_RMSE"]]
                 .rename(columns={"GroupCV_R2": "R²", "GroupCV_RMSE": "RMSE (years)"}),
                 hide_index=True)

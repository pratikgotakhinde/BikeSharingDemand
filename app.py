import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Bike Sharing Demand Dashboard",
    layout="wide",
    page_icon="🚲"
)

# Custom CSS - Dark theme with bike-green accents
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top left, #222831 0%, #0f141a 40%, #000000 100%);
    color: #f5f5f5;
    font-family: "Inter", system-ui, sans-serif;
}
.block-container {
    padding-top: 1rem;
}
h1, h2, h3 {
    font-family: "Poppins", "Inter", sans-serif;
    color: #e8f9fd;
}
[data-testid="stSidebar"] {
    background: rgba(10, 16, 24, 0.95);
    border-right: 1px solid #1f2933;
}
.card {
    background: linear-gradient(135deg, #111827, #020617);
    padding: 16px 24px;
    border-radius: 16px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
    border: 1px solid rgba(148, 163, 184, 0.3);
    margin-bottom: 1rem;
}
[data-testid="stMetricValue"] {
    color: #4ade80;
    font-weight: 600;
}
.stButton>button {
    border-radius: 999px;
    background: linear-gradient(90deg, #22c55e, #a3e635);
    color: #020617;
    border: none;
    font-weight: 600;
}
.stButton>button:hover {
    filter: brightness(1.05);
}
.chart-caption {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-top: -6px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<h1 style='text-align:center; margin-bottom:0;'>
    🚲 Bike Sharing Demand Dashboard
</h1>
<p style='text-align:center; color:#9ca3af; font-size:0.95rem;'>
    Explore how time, season and weather shape bike rental behaviour in Washington, D.C.
</p>
<hr style="border-color:#1f2933;">
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["hour"] = df["datetime"].dt.hour
    
    season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
    df["season_name"] = df["season"].map(season_map)
    
    def get_day_period(h):
        if h < 6:
            return "night"
        elif h < 12:
            return "morning"
        elif h < 18:
            return "afternoon"
        else:
            return "evening"
    
    df["day_period"] = df["hour"].apply(get_day_period)
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

year_opt = st.sidebar.selectbox("Year", ["All", 2011, 2012])

all_seasons = sorted(df["season_name"].dropna().unique())
season_opt = st.sidebar.multiselect(
    "Season",
    options=["All"] + all_seasons,
    default=["All"]
)
if "All" in season_opt or not season_opt:
    season_opt = all_seasons

all_weather = sorted(df["weather"].dropna().unique())
weather_opt = st.sidebar.multiselect(
    "Weather category",
    options=["All"] + all_weather,
    default=["All"]
)
if "All" in weather_opt or not weather_opt:
    weather_opt = all_weather

workingday_map = {"All": None, "Working days only": 1, "Non-working days only": 0}
workingday_label = st.sidebar.selectbox("Working day filter", list(workingday_map.keys()))

min_hour, max_hour = st.sidebar.slider("Hour range", 0, 23, (0, 23))

target_choice = st.sidebar.selectbox("Show", ["Total rentals", "Registered users"])
target_col = "registered" if target_choice == "Registered users" else "count"

# Apply filters
df_filtered = df.copy()

if year_opt != "All":
    df_filtered = df_filtered[df_filtered["year"] == year_opt]

df_filtered = df_filtered[df_filtered["season_name"].isin(season_opt)]
df_filtered = df_filtered[df_filtered["weather"].isin(weather_opt)]
df_filtered = df_filtered[(df_filtered["hour"] >= min_hour) & (df_filtered["hour"] <= max_hour)]

wd_val = workingday_map[workingday_label]
if wd_val is not None:
    df_filtered = df_filtered[df_filtered["workingday"] == wd_val]

# KPIs
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    with col_kpi1:
        st.metric(f"Total {target_choice.lower()} (filtered)", f"{df_filtered[target_col].sum():,.0f}")
    
    with col_kpi2:
        st.metric(f"Average {target_choice.lower()} per hour", f"{df_filtered[target_col].mean():.1f}")
    
    with col_kpi3:
        if not df_filtered.empty:
            peak_hour = df_filtered.groupby("hour")[target_col].mean().idxmax()
        else:
            peak_hour = "-"
        st.metric("Peak hour (filtered)", f"{peak_hour}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Time patterns", "Season & Weather", "Correlations"])

# Tab 1: Time patterns
with tab1:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Mean rentals by hour")
            fig, ax = plt.subplots()
            sns.lineplot(data=df_filtered, x="hour", y=target_col, estimator=np.mean, ci=95, marker="o", ax=ax, color="#22c55e")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Mean rentals")
            ax.grid(alpha=0.25)
            st.pyplot(fig)
            st.markdown("<p class='chart-caption'>Typical commuter peaks around morning and evening hours.</p>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("Mean rentals by period of day")
            fig2, ax2 = plt.subplots()
            order = ["night", "morning", "afternoon", "evening"]
            sns.barplot(data=df_filtered, x="day_period", y=target_col, estimator=np.mean, ci=95, order=order, palette="Greens")
            ax2.set_xlabel("Period of day")
            ax2.set_ylabel("Mean rentals")
            st.pyplot(fig2)
            st.markdown("<p class='chart-caption'>Evening and morning windows highlight strong rush-hour usage.</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Hourly rentals by day of week")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df_filtered, x="hour", y=target_col, hue="day_of_week", estimator=np.mean, ci=None, marker="o", ax=ax3)
        ax3.set_xlabel("Hour")
        ax3.set_ylabel("Mean rentals")
        ax3.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        ax3.grid(alpha=0.25)
        st.pyplot(fig3)
        st.markdown("<p class='chart-caption'>Compare workdays vs weekend profiles to see commuting effects.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Season & Weather
with tab2:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col4, col5 = st.columns(2)
        
        with col4:
            st.subheader("Mean rentals by season")
            fig4, ax4 = plt.subplots()
            sns.barplot(data=df_filtered, x="season_name", y=target_col, estimator=np.mean, ci=95, palette="YlGn", ax=ax4)
            ax4.set_xlabel("Season")
            ax4.set_ylabel("Mean rentals")
            st.pyplot(fig4)
            st.markdown("<p class='chart-caption'>Warm seasons boost usage; winter typically shows a drop.</p>", unsafe_allow_html=True)
        
        with col5:
            st.subheader("Mean rentals by weather")
            fig5, ax5 = plt.subplots()
            sns.barplot(data=df_filtered, x="weather", y=target_col, estimator=np.mean, ci=95, palette="GnBu", ax=ax5)
            ax5.set_xlabel("Weather category")
            ax5.set_ylabel("Mean rentals")
            st.pyplot(fig5)
            st.markdown("<p class='chart-caption'>Clear days drive more rides; harsh conditions dampen demand.</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Tab 3: Correlations
with tab3:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Correlation heatmap (numeric features)")
        
        num_cols = df_filtered.select_dtypes(include=["int64", "float64"]).columns
        
        if len(num_cols) > 1 and not df_filtered.empty:
            corr = df_filtered[num_cols].corr()
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap="rocket_r", ax=ax6)
            st.pyplot(fig6)
            st.markdown("<p class='chart-caption'>Check how temperature, humidity and other factors move with demand.</p>", unsafe_allow_html=True)
        else:
            st.info("Not enough numeric data after filtering to compute correlations.")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style='margin-top: 3rem; border: none; border-top: 1px solid #333;'>
<p style='text-align: center; color: #9ca3af; font-size: 0.9rem; padding: 1rem 0;'>
    Created by <strong style='color: #4ade80;'>Pratik Gotakhinde</strong>
</p>
""", unsafe_allow_html=True)


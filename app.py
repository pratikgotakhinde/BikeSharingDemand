import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
            # Group data by hour and calculate mean with confidence interval
            hourly_data = df_filtered.groupby("hour")[target_col].agg(['mean', 'std', 'count']).reset_index()
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=hourly_data["hour"],
                y=hourly_data["mean"],
                mode='lines+markers',
                name='Mean rentals',
                line=dict(color='#22c55e', width=3),
                marker=dict(size=8),
                hovertemplate='<b>Hour:</b> %{x}<br><b>Mean rentals:</b> %{y:.1f}<extra></extra>'
            ))
            
            fig1.update_layout(
                xaxis_title="Hour of day",
                yaxis_title="Mean rentals",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f5f5f5'),
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("<p class='chart-caption'>Typical commuter peaks around morning and evening hours.</p>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("Mean rentals by period of day")
            # Calculate mean by period
            period_order = ["night", "morning", "afternoon", "evening"]
            period_data = df_filtered.groupby("day_period")[target_col].mean().reindex(period_order).reset_index()
            
            fig2 = px.bar(
                period_data,
                x="day_period",
                y=target_col,
                color="day_period",
                color_discrete_map={
                    "night": "#065f46",
                    "morning": "#16a34a",
                    "afternoon": "#22c55e",
                    "evening": "#4ade80"
                }
            )
            
            fig2.update_traces(
                hovertemplate='<b>Period:</b> %{x}<br><b>Mean rentals:</b> %{y:.1f}<extra></extra>'
            )
            
            fig2.update_layout(
                xaxis_title="Period of day",
                yaxis_title="Mean rentals",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f5f5f5'),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<p class='chart-caption'>Evening and morning windows highlight strong rush-hour usage.</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Hourly rentals by day of week")
        
        # Group by hour and day_of_week
        day_hour_data = df_filtered.groupby(["hour", "day_of_week"])[target_col].mean().reset_index()
        
        fig3 = px.line(
            day_hour_data,
            x="hour",
            y=target_col,
            color="day_of_week",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig3.update_traces(
            hovertemplate='<b>%{fullData.name}</b><br>Hour: %{x}<br>Mean rentals: %{y:.1f}<extra></extra>'
        )
        
        fig3.update_layout(
            xaxis_title="Hour",
            yaxis_title="Mean rentals",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f5f5f5'),
            legend_title="Day of week",
            height=450
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("<p class='chart-caption'>Compare workdays vs weekend profiles to see commuting effects.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Season & Weather
with tab2:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col4, col5 = st.columns(2)
        
        with col4:
            st.subheader("Mean rentals by season")
            season_data = df_filtered.groupby("season_name")[target_col].mean().reset_index()
            season_data = season_data.sort_values(target_col, ascending=False)
            
            fig4 = px.bar(
                season_data,
                x="season_name",
                y=target_col,
                color="season_name",
                color_discrete_map={
                    "spring": "#84cc16",
                    "summer": "#22c55e",
                    "fall": "#fb923c",
                    "winter": "#60a5fa"
                }
            )
            
            fig4.update_traces(
                hovertemplate='<b>Season:</b> %{x}<br><b>Mean rentals:</b> %{y:.1f}<extra></extra>'
            )
            
            fig4.update_layout(
                xaxis_title="Season",
                yaxis_title="Mean rentals",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f5f5f5'),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown("<p class='chart-caption'>Warm seasons boost usage; winter typically shows a drop.</p>", unsafe_allow_html=True)
        
        with col5:
            st.subheader("Mean rentals by weather")
            weather_data = df_filtered.groupby("weather")[target_col].mean().reset_index()
            weather_data = weather_data.sort_values(target_col, ascending=False)
            
            fig5 = px.bar(
                weather_data,
                x="weather",
                y=target_col,
                color="weather",
                color_discrete_sequence=['#22c55e', '#fbbf24', '#f87171', '#dc2626']
            )
            
            fig5.update_traces(
                hovertemplate='<b>Weather:</b> %{x}<br><b>Mean rentals:</b> %{y:.1f}<extra></extra>'
            )
            
            fig5.update_layout(
                xaxis_title="Weather category",
                yaxis_title="Mean rentals",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f5f5f5'),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig5, use_container_width=True)
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
            
            fig6 = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale='RdBu_r',
                zmid=0,
                text=corr.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig6.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f5f5f5'),
                height=600,
                xaxis=dict(side='bottom')
            )
            st.plotly_chart(fig6, use_container_width=True)
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

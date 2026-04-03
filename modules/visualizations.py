"""
modules/visualizations.py
Plotly-based crime heatmaps, hotspot maps, and trend analytics.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

DARK = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0e1220",
    font=dict(color="#94a3b8", family="monospace", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ──────────────────────────────────────────────────────────────────────────
# HEATMAP / DENSITY MAP
# ──────────────────────────────────────────────────────────────────────────
def fig_density_heatmap(df: pd.DataFrame,
                         lat_col: str = "Latitude",
                         lon_col: str = "Longitude",
                         color_col: str | None = None,
                         title: str = "Crime Density Heatmap") -> go.Figure:
    """Mapbox-free density heatmap using scatter with contour."""
    sub = df[[lat_col, lon_col]].dropna().sample(min(len(df), 5000), random_state=42)

    fig = go.Figure(go.Densitymapbox(
        lat=sub[lat_col],
        lon=sub[lon_col],
        radius=14,
        colorscale=[
            [0.00, "#080b12"],
            [0.25, "#00468b"],
            [0.55, "#00c8e8"],
            [0.75, "#f4a261"],
            [0.90, "#e63946"],
            [1.00, "#ff1744"],
        ],
        showscale=True,
        colorbar=dict(title="Density", tickfont=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        opacity=0.8,
    ))
    center_lat = sub[lat_col].mean()
    center_lon = sub[lon_col].mean()
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        mapbox_center={"lat": center_lat, "lon": center_lon},
        mapbox_zoom=11,
        title=title,
        height=480,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="monospace"),
    )
    return fig


def fig_scatter_map(df: pd.DataFrame,
                     lat_col: str = "Latitude",
                     lon_col: str = "Longitude",
                     color_col: str = "Primary Type",
                     title: str = "Crime Incidents Map") -> go.Figure:
    """Scatter map coloured by crime type."""
    sub = df.dropna(subset=[lat_col, lon_col]).sample(min(len(df), 3000), random_state=42)
    fig = px.scatter_mapbox(
        sub, lat=lat_col, lon=lon_col,
        color=color_col,
        hover_data={"Date": True, "District": True},
        zoom=11,
        height=480,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="monospace"),
        legend=dict(bgcolor="rgba(14,18,32,0.8)", bordercolor="#252d45"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────
# TREND CHARTS
# ──────────────────────────────────────────────────────────────────────────
def fig_hourly_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of incidents by hour."""
    hourly = df.groupby("Hour").size().reset_index(name="count")
    fig = go.Figure(go.Bar(
        x=hourly["Hour"], y=hourly["count"],
        marker=dict(
            color=hourly["count"],
            colorscale=[[0,"#0e4d8a"],[0.5,"#00c8e8"],[1,"#e63946"]],
            showscale=False,
        ),
        text=hourly["count"], textposition="outside",
    ))
    fig.update_layout(
        title="Incidents by Hour of Day",
        xaxis=dict(title="Hour", dtick=2, gridcolor="#1e2a40"),
        yaxis=dict(title="Count", gridcolor="#1e2a40"),
        **DARK,
    )
    return fig


def fig_weekly_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of daily incidents over time."""
    daily = (df.groupby(df["Date"].dt.date).size()
               .reset_index(name="count")
               .rename(columns={"Date":"date"}))
    daily["date"] = pd.to_datetime(daily["date"])
    # Rolling 7-day average
    daily["rolling7"] = daily["count"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["count"],
        mode="lines", name="Daily",
        line=dict(color="rgba(0,200,232,0.3)", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily["rolling7"],
        mode="lines", name="7-Day Avg",
        line=dict(color="#00c8e8", width=2),
        fill="tozeroy", fillcolor="rgba(0,200,232,0.05)",
    ))
    fig.update_layout(
        title="Daily Incident Trend (with 7-day rolling avg)",
        xaxis=dict(title="Date", gridcolor="#1e2a40"),
        yaxis=dict(title="Incidents", gridcolor="#1e2a40"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **DARK,
    )
    return fig


def fig_crime_type_bar(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    top = df["Primary Type"].value_counts().head(top_n)
    colors = px.colors.sequential.Reds_r[:top_n]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index,
        orientation="h",
        marker_color="#e63946",
        text=top.values, textposition="auto",
    ))
    fig.update_layout(
        title=f"Top {top_n} Crime Types",
        xaxis=dict(title="Count", gridcolor="#1e2a40"),
        yaxis=dict(autorange="reversed", gridcolor="#1e2a40"),
        **DARK,
    )
    return fig


def fig_district_breakdown(df: pd.DataFrame) -> go.Figure:
    dist = df.groupby("District").size().reset_index(name="count").sort_values("count", ascending=False).head(15)
    fig = go.Figure(go.Bar(
        x=dist["District"].astype(str),
        y=dist["count"],
        marker=dict(color=dist["count"], colorscale=[[0,"#00468b"],[1,"#e63946"]], showscale=False),
        text=dist["count"], textposition="auto",
    ))
    fig.update_layout(
        title="Incidents by Police District (Top 15)",
        xaxis=dict(title="District", gridcolor="#1e2a40"),
        yaxis=dict(title="Incidents", gridcolor="#1e2a40"),
        **DARK,
    )
    return fig


def fig_arrest_rate(df: pd.DataFrame) -> go.Figure:
    rate = df.groupby("Primary Type")["Arrest"].mean().sort_values(ascending=False).head(10)
    fig = go.Figure(go.Bar(
        y=rate.index, x=rate.values * 100,
        orientation="h",
        marker_color="#4ade80",
        text=[f"{v:.1f}%" for v in rate.values * 100],
        textposition="auto",
    ))
    fig.update_layout(
        title="Arrest Rate by Crime Type (Top 10)",
        xaxis=dict(title="Arrest Rate (%)", gridcolor="#1e2a40"),
        yaxis=dict(autorange="reversed", gridcolor="#1e2a40"),
        **DARK,
    )
    return fig


def fig_dayofweek_heatmap(df: pd.DataFrame) -> go.Figure:
    """Hour × Day-of-week heatmap."""
    pivot = df.groupby(["DayOfWeek", "Hour"]).size().unstack(fill_value=0)
    day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(range(24)),
        y=[day_labels[i] for i in pivot.index],
        colorscale=[
            [0.0, "#080b12"],
            [0.4, "#00468b"],
            [0.7, "#f4a261"],
            [1.0, "#e63946"],
        ],
        showscale=True,
        colorbar=dict(tickfont=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
    ))
    fig.update_layout(
        title="Crime Activity: Hour × Day of Week",
        xaxis=dict(title="Hour", dtick=2, gridcolor="#1e2a40"),
        yaxis=dict(title="Day", gridcolor="#1e2a40"),
        **DARK,
    )
    return fig


def fig_monthly_trend(df: pd.DataFrame) -> go.Figure:
    monthly = df.groupby(["Year","Month"]).size().reset_index(name="count")
    monthly["period"] = pd.to_datetime(monthly[["Year","Month"]].assign(Day=1))
    fig = go.Figure()
    for yr in sorted(monthly["Year"].unique()):
        sub = monthly[monthly["Year"]==yr].sort_values("Month")
        fig.add_trace(go.Scatter(
            x=sub["Month"], y=sub["count"],
            mode="lines+markers", name=str(yr),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title="Monthly Incident Count by Year",
        xaxis=dict(title="Month", dtick=1, gridcolor="#1e2a40",
                   ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"],
                   tickvals=list(range(1,13))),
        yaxis=dict(title="Incidents", gridcolor="#1e2a40"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        **DARK,
    )
    return fig

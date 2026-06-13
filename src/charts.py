import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, List

def create_comparison_bar_chart(metrics_df: pd.DataFrame, y_column: str, title: str, y_label: str):
    """Creates a bar chart comparing baseline and optimized models for a specific metric."""
    fig = px.bar(
        metrics_df,
        x="Model",
        y=y_column,
        color="Model",
        title=title,
        labels={y_column: y_label},
        text_auto='.2f',
        color_discrete_map={"Baseline": "#636EFA", "Optimized": "#00CC96"}
    )
    return fig

def create_rouge_comparison_chart(rouge_df: pd.DataFrame):
    """Creates a grouped bar chart for ROUGE scores."""
    fig = px.bar(
        rouge_df,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="ROUGE Score Comparison",
        text_auto='.3f',
        color_discrete_map={"Baseline": "#636EFA", "Optimized": "#00CC96"}
    )
    return fig

def create_tradeoff_scatter_plot(comparison_data: List[Dict]):
    """Creates an accuracy/efficiency trade-off scatter plot."""
    df = pd.DataFrame(comparison_data)

    if "Quality" not in df.columns:
        df["Quality"] = 0.75

    fig = px.scatter(
        df,
        x="Energy (J)",
        y="Quality",
        color="Model",
        size="Memory (MB)",
        hover_name="Model",
        title="Accuracy vs Efficiency Trade-off",
        labels={
            "Energy (J)": "Energy Consumption (J)",
            "Memory (MB)": "Memory Usage (MB)",
            "Quality": "Summary Quality (Proxy/ROUGE-L)",
        },
        color_discrete_map={"Baseline": "#636EFA", "Optimized": "#00CC96"}
    )
    return fig

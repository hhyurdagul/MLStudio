import altair as alt
import polars as pl
import streamlit as st

from mlstudio.backend.types import (
    CrossValidationMetrics,
    GridSearchSummary,
    RegressionMetrics,
)


def render_metrics(metrics: RegressionMetrics) -> None:
    columns = st.columns(4)
    columns[0].metric("R²", f"{metrics.r2:.4f}")
    columns[1].metric("MAE", f"{metrics.mae:.4f}")
    columns[2].metric("RMSE", f"{metrics.rmse:.4f}")
    columns[3].metric(
        "MAPE",
        "N/A" if metrics.mape is None else f"{metrics.mape:.2f}%",
    )
    if metrics.mape is None:
        st.caption("MAPE is unavailable because every actual target value is zero.")


def render_cross_validation_metrics(metrics: CrossValidationMetrics) -> None:
    columns = st.columns(4)
    columns[0].metric("R²", f"{metrics.r2_mean:.4f} ± {metrics.r2_std:.4f}")
    columns[1].metric("MAE", f"{metrics.mae_mean:.4f} ± {metrics.mae_std:.4f}")
    columns[2].metric("RMSE", f"{metrics.rmse_mean:.4f} ± {metrics.rmse_std:.4f}")
    columns[3].metric(
        "MAPE",
        (
            "N/A"
            if metrics.mape_mean is None or metrics.mape_std is None
            else f"{metrics.mape_mean:.2f}% ± {metrics.mape_std:.2f}%"
        ),
    )


def render_grid_search(summary: GridSearchSummary | None) -> None:
    if summary is None:
        return
    st.write("Best grid-search parameters")
    st.json(summary.best_parameters)
    st.metric("Best cross-validation R²", f"{summary.best_score:.4f}")


def render_predictions(data: pl.DataFrame, *, key: str) -> None:
    data_tab, graph_tab = st.tabs(["Data", "Graph"])
    with data_tab:
        st.dataframe(data, hide_index=True)
        st.download_button(
            "Download predictions",
            data=data.write_csv().encode(),
            file_name="predictions.csv",
            mime="text/csv",
            key=key,
            on_click="ignore",
        )
    with graph_tab:
        chart_data = (
            data.with_row_index("Row")
            .unpivot(index="Row", variable_name="Series", value_name="Value")
            .to_pandas()
        )
        chart = (
            alt.Chart(chart_data)
            .mark_line()
            .encode(
                x=alt.X("Row:Q", title="Row"),
                y=alt.Y("Value:Q", title="Value"),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(
                        domain=["Real", "Prediction"],
                        range=["#1f77b4", "#ff7f0e"],
                    ),
                    legend=alt.Legend(title=None),
                ),
                tooltip=["Row:Q", "Series:N", "Value:Q"],
            )
        )
        st.altair_chart(chart, width="stretch")

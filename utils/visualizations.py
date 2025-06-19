"""Plotting utilities used by the web application."""

from typing import Iterable

import pandas as pd
import plotly.graph_objects as go

def create_return_plot(
    dates: Iterable[pd.Timestamp],
    naive_series: pd.Series,
    strategy_series: pd.Series,
) -> str:
    """Return an HTML snippet with cumulative-return lines.

    Parameters
    ----------
    dates : Iterable[pd.Timestamp]
        Dates for the x-axis.
    naive_series : pd.Series
        Cumulative returns of the buy-and-hold benchmark.
    strategy_series : pd.Series
        Cumulative returns of the evaluated strategy.

    Returns
    -------
    str
        HTML representation of the Plotly figure.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=naive_series, mode='lines', name='Naive (Buy & Hold) Return',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=strategy_series, mode='lines', name='Market Timing Strategy Return',
        line=dict(color='red')
    ))
    fig.update_layout(
        title='Cumulative Returns Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Return'
    )
    fig.update_xaxes(type='date')
    return fig.to_html(full_html=False)

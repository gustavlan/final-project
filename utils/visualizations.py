import plotly.graph_objects as go

def create_return_plot(dates, naive_series, strategy_series):
    """Generate an interactive Plotly line chart comparing naive and strategy cumulative returns."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=naive_series, mode='lines', name='Naive (Buy & Hold) Return',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=strategy_series, mode='lines', name='Selected Strategy Return',
        line=dict(color='red')
    ))
    fig.update_layout(
        title='Cumulative Returns Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Return'
    )
    fig.update_xaxes(type='date')
    return fig.to_html(full_html=False)

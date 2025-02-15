import plotly.graph_objects as go

def create_return_plot(dates, returns):
    """Generate an interactive Plotly line chart for cumulative returns over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=returns, mode='lines', name='Cumulative Returns'))
    fig.update_layout(
        title='Cumulative Returns Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Return'
    )
    # Ensure x-axis is treated as dates
    fig.update_xaxes(type='date')
    return fig.to_html(full_html=False)

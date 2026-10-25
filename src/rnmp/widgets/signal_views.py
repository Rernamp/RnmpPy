import plotly.graph_objects as go
import numpy as np

def plot_signals(time, signal):
    is_single_array = signal.ndim == 1
    channels_count = 1 if is_single_array else signal.shape[1]
    _date = []
    for _i in range(0, channels_count):
        _date.append(go.Scatter(x=time, y=(signal if is_single_array else signal[:,_i]), name=f"upsampled_signal_ch_{_i}", mode="lines"))

    return go.Figure(
        data=_date
    )
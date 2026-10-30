"""Plotly views of the DOA estimates: the MUSIC spectrum on the sphere of directions, static or animated."""

import numpy as np
import plotly.graph_objects as go

from rnmp.doa.music import direction_vectors


def _surface_color(spectrum_dB, vmin):
    """Attenuation below the maximum in whole dB, 0 .. -vmin (vmin >= -60), as uint8, with the first azimuth
    repeated to close the seam at +-180.

    The array goes to the browser as base64 (bdata); bytes below 64 encode without '/' and '+', which the
    marimo HTML output would escape to 6 characters each (negative int8 values are full of them).
    """
    closed = np.concatenate([spectrum_dB, spectrum_dB[:, :1]], axis=1)
    return np.clip(-closed, 0, -vmin).round().astype(np.uint8)


def _colorbar(vmin):
    """Colorbar for the attenuation surface, labeled in dB relative to the maximum (0, -5, -10, ...)."""
    ticks = np.arange(0, -vmin + 1, 5 if vmin >= -30 else 10)
    return dict(title="dB", tickvals=ticks, ticktext=[f"{-t:g}" for t in ticks])


def _peak_points(peaks, count):
    """Ray and marker coordinates and the label per peak, `count` of them: missing peaks are empty, so every
    animation frame has the same traces."""
    points = []
    for k in range(count):
        if k < len(peaks):
            p_az, p_el = peaks[k]
            u = direction_vectors(np.array([p_az]), np.array([p_el]))[0, 0]
            points.append(([[0, 1.3 * c] for c in u], [[1.02 * c] for c in u], [f"{p_az:.0f}°, {p_el:.0f}°"]))
        else:
            points.append(([[], [], []], [[], [], []], []))
    return points


def _peak_frame_traces(peaks, count):
    """Only what changes between frames; the style stays from the traces of _peak_traces."""
    traces = []
    for ray, dot, label in _peak_points(peaks, count):
        traces.append(go.Scatter3d(x=ray[0], y=ray[1], z=ray[2]))
        traces.append(go.Scatter3d(x=dot[0], y=dot[1], z=dot[2], text=label))
    return traces


def _peak_traces(peaks, count):
    """Ray + marker per peak."""
    traces = []
    for k, (ray, dot, label) in enumerate(_peak_points(peaks, count)):
        traces.append(go.Scatter3d(
            x=ray[0], y=ray[1], z=ray[2], mode="lines",
            line=dict(color="red", width=6), showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Scatter3d(
            x=dot[0], y=dot[1], z=dot[2], mode="markers+text",
            marker=dict(size=9, color="red", line=dict(color="white", width=2)),
            text=label, textposition="top center", textfont=dict(color="red", size=14),
            name=f"peak {k + 1}", hovertemplate="%{text}<extra></extra>",
        ))
    return traces


def _sphere(spectrum_dB, azimuths_deg, elevations_deg, peaks, peak_count, vmin, title):
    u = direction_vectors(np.append(azimuths_deg, azimuths_deg[0] + 360), elevations_deg)
    fig = go.Figure([
        go.Surface(x=u[..., 0], y=u[..., 1], z=u[..., 2], surfacecolor=_surface_color(spectrum_dB, vmin),
                   colorscale="Viridis", reversescale=True, cmin=0, cmax=-vmin, colorbar=_colorbar(vmin),
                   hoverinfo="skip"),
        *_peak_traces(peaks, peak_count),
        # reference directions: 0 deg = toward the user, 90 deg = right
        go.Scatter3d(x=[0, 1.25, 0, -1.25], y=[-1.25, 0, 1.25, 0], z=[0, 0, 0, 0], mode="text",
                     text=["0° user", "90°", "180°", "-90°"], textfont=dict(color="gray"),
                     showlegend=False, hoverinfo="skip"),
    ])

    axis = dict(range=[-1.35, 1.35], showticklabels=False)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="x, right", **axis),
            yaxis=dict(title="y, away", **axis),
            zaxis=dict(title="z, up", showticklabels=False, range=[-0.1, 1.35]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.45 / 2.7),
            camera=dict(eye=dict(x=1.2, y=-1.6, z=1.0)),
        ),
        uirevision="sphere",  # keep the camera when the figure is updated
        height=700,
    )
    return fig


def sphere_figure(spectrum_dB, azimuths_deg, elevations_deg, peaks=(), vmin=-30.0, title="MUSIC spectrum (sphere)"):
    """Spectrum (n_el, n_az), dB, as a heat map on the upper hemisphere of directions; peaks [(az, el), ...]
    are marked. Axes as in doa.py: x -> right, y -> away from the user, z -> up."""
    return _sphere(spectrum_dB, azimuths_deg, elevations_deg, peaks, len(peaks), vmin, title)


def sphere_animation(spectra_dB, azimuths_deg, elevations_deg, peaks, labels, titles, vmin=-30.0, frame_ms=150):
    """sphere_figure animated over windows: spectra_dB (W, n_el, n_az), peaks / labels / titles - one per window.

    The sphere is sent to the browser once; a frame carries only the surface color (whole dB), the peaks and
    the title, so the plotly slider and the play button run in the browser without going back to Python.
    """
    peak_count = max((len(p) for p in peaks), default=0)
    fig = _sphere(spectra_dB[0], azimuths_deg, elevations_deg, peaks[0], peak_count, vmin, titles[0])

    updated_traces = list(range(1 + 2 * peak_count))  # surface + peak rays/markers, not the reference labels
    fig.frames = [
        go.Frame(
            name=str(i),
            data=[
                # a uint8 array goes out as compact base64 (bdata) and is validated far faster than a list
                go.Surface(surfacecolor=_surface_color(spectra_dB[i], vmin)),
                *_peak_frame_traces(peaks[i], peak_count),
            ],
            traces=updated_traces,
            layout=go.Layout(title_text=titles[i]),
        )
        for i in range(len(spectra_dB))
    ]

    def play(duration):
        return [None, dict(frame=dict(duration=duration, redraw=True), transition=dict(duration=0), fromcurrent=True)]

    fig.update_layout(
        margin=dict(b=120),  # room for the play buttons and the slider under the scene
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False, x=0, y=0, xanchor="left", yanchor="top",
            pad=dict(t=40, r=10),
            buttons=[
                dict(label="▶", method="animate", args=play(frame_ms)),
                dict(label="▶▶", method="animate", args=play(frame_ms // 4)),
                dict(label="❚❚", method="animate",
                     args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.22, len=0.78, y=0, yanchor="top", pad=dict(t=40),
            bgcolor="#d0d0d0", activebgcolor="#ff4136", bordercolor="#888", borderwidth=1, ticklen=0, minorticklen=0,
            currentvalue=dict(prefix="t = ", suffix=" s", visible=True, font=dict(size=14, color="#444")),
            # too many steps for tick labels: hide them; currentvalue has its own font, the current one is shown
            font=dict(color="rgba(0, 0, 0, 0)"),
            steps=[
                dict(method="animate", label=labels[i],
                     args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                          transition=dict(duration=0))])
                for i in range(len(spectra_dB))
            ],
        )],
    )
    return fig

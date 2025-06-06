import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, LogFormatter
import matplotlib.font_manager as font_manager

st.set_page_config(layout='wide', page_title="CSV Plotter")

st.title("CSV Plotter and Grapher (Advanced Log Axis Handling)")

# File upload
csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.success("CSV loaded!")
    st.dataframe(df.head())

    columns = list(df.columns)
    if len(columns) < 2:
        st.warning("CSV must have at least two columns.")
        st.stop()

    # ---- Sidebar for controls ----
    with st.sidebar:
        st.header("Plot Settings")

        x_col = st.selectbox("X column", columns)
        y_col = st.selectbox("Y column", columns, index=1 if len(columns) > 1 else 0)

        plot_type = st.selectbox("Plot Type", ["Line", "Scatter", "Line + Scatter"])

        st.markdown("---")
        st.subheader("Axis Scales (choose separately)")
        x_scale = st.selectbox("X Axis Scale", ["Linear", "Logarithmic", "Probability"])
        y_scale = st.selectbox("Y Axis Scale", ["Linear", "Logarithmic", "Probability"])

        # Axis labels
        st.markdown("---")
        st.subheader("Axis Labels")
        x_label = st.text_input("X Axis Label", value=x_col)
        y_label = st.text_input("Y Axis Label", value=y_col)

        # Font
        font_list = sorted(set(f.name for f in font_manager.fontManager.ttflist))
        font_default = font_list.index("DejaVu Sans") if "DejaVu Sans" in font_list else 0
        font_family = st.selectbox("Font Family", font_list, index=font_default)
        font_size = st.slider("Font size", 8, 32, 14)
        font_weight = st.checkbox("Bold Axis Labels", False)

        # Data preparation for axis range selection
        def get_log10_range(series):
            finite = series.replace([np.inf, -np.inf], np.nan).dropna()
            finite = finite[finite > 0]
            if finite.empty:
                return None, None
            return np.log10(finite.min()), np.log10(finite.max())

        x_data = df[x_col].replace([np.inf, -np.inf], np.nan).dropna()
        y_data = df[y_col].replace([np.inf, -np.inf], np.nan).dropna()
        x_lin_min, x_lin_max = float(x_data.min()), float(x_data.max())
        y_lin_min, y_lin_max = float(y_data.min()), float(y_data.max())
        x_log_min, x_log_max = get_log10_range(x_data)
        y_log_min, y_log_max = get_log10_range(y_data)

        st.markdown("---")
        st.subheader("Axis Range and Step")

        # X axis controls
        st.markdown("**X Axis**")
        if x_scale == "Logarithmic" and x_log_min is not None:
            st.caption(f"X (log10) range in data: {x_log_min:.3g} to {x_log_max:.3g} (corresponds to {10**x_log_min:.3g} to {10**x_log_max:.3g})")
            xm_log = st.number_input("X min (log10)", value=float(x_log_min), key="x_min_log")
            xM_log = st.number_input("X max (log10)", value=float(x_log_max), key="x_max_log")
            xs_log = st.number_input("X step (log10)", value=max((x_log_max-x_log_min)/10,0.1), min_value=1e-8, format="%.4g", key="x_step_log")
        else:
            if x_scale == "Logarithmic":
                st.warning("X data does not have positive values for log scale.")
            st.caption(f"X range in data: {x_lin_min:.4g} to {x_lin_max:.4g}")
            xm_lin = st.number_input("X min", value=x_lin_min, key="x_min")
            xM_lin = st.number_input("X max", value=x_lin_max, key="x_max")
            xs_lin = st.number_input("X step", value=max((x_lin_max-x_lin_min)/10, 1e-6), min_value=1e-8, format="%.4g", key="x_step")

        # Y axis controls
        st.markdown("**Y Axis**")
        if y_scale == "Logarithmic" and y_log_min is not None:
            st.caption(f"Y (log10) range in data: {y_log_min:.3g} to {y_log_max:.3g} (corresponds to {10**y_log_min:.3g} to {10**y_log_max:.3g})")
            ym_log = st.number_input("Y min (log10)", value=float(y_log_min), key="y_min_log")
            yM_log = st.number_input("Y max (log10)", value=float(y_log_max), key="y_max_log")
            ys_log = st.number_input("Y step (log10)", value=max((y_log_max-y_log_min)/10,0.1), min_value=1e-8, format="%.4g", key="y_step_log")
        else:
            if y_scale == "Logarithmic":
                st.warning("Y data does not have positive values for log scale.")
            st.caption(f"Y range in data: {y_lin_min:.4g} to {y_lin_max:.4g}")
            ym_lin = st.number_input("Y min", value=y_lin_min, key="y_min")
            yM_lin = st.number_input("Y max", value=y_lin_max, key="y_max")
            ys_lin = st.number_input("Y step", value=max((y_lin_max-y_lin_min)/10, 1e-6), min_value=1e-8, format="%.4g", key="y_step")

        st.markdown("---")
        st.subheader("Tick Label Format")
        tick_format = st.selectbox("Tick Label Format", ["Default", "Scientific", "Decimal"])

        st.markdown("---")
        st.subheader("Colors and Style")
        scatter_color = st.color_picker("Scatter color", "#1f77b4")
        line_color = st.color_picker("Line color", "#ff7f0e")
        line_thickness = st.slider("Line thickness", 1, 10, 2)
        border_thickness = st.slider("Border thickness", 1, 10, 2)
        show_legend = st.checkbox("Show Legend", True)

        st.markdown("---")
        st.subheader("Plot Size")
        plot_width = st.slider("Plot width (inches)", 4, 16, 8)
        plot_height = st.slider("Plot height (inches)", 3, 10, 5)

    # ---- Main area for live plot ----
    try:
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        x = df[x_col]
        y = df[y_col]

        # Font dictionary
        fontdict = {"fontsize": font_size, "fontweight": "bold" if font_weight else "normal", "fontname": font_family}

        # X Axis scaling and ticks
        if x_scale == "Logarithmic" and x_log_min is not None:
            ax.set_xscale("log")
            x_ticks = 10 ** np.arange(xm_log, xM_log+xs_log, xs_log)
            ax.set_xlim(10 ** xm_log, 10 ** xM_log)
            ax.set_xticks(x_ticks)
        else:
            if x_scale == "Logarithmic":
                st.warning("Cannot plot X axis in log scale due to non-positive values.")
            ax.set_xlim(xm_lin, xM_lin)
            ax.set_xticks(np.arange(xm_lin, xM_lin+xs_lin, xs_lin))

        # Y Axis scaling and ticks
        if y_scale == "Logarithmic" and y_log_min is not None:
            ax.set_yscale("log")
            y_ticks = 10 ** np.arange(ym_log, yM_log+ys_log, ys_log)
            ax.set_ylim(10 ** ym_log, 10 ** yM_log)
            ax.set_yticks(y_ticks)
        else:
            if y_scale == "Logarithmic":
                st.warning("Cannot plot Y axis in log scale due to non-positive values.")
            ax.set_ylim(ym_lin, yM_lin)
            ax.set_yticks(np.arange(ym_lin, yM_lin+ys_lin, ys_lin))

        # Probability scale (Y axis only, for demonstration)
        if y_scale == "Probability":
            from scipy.stats import norm
            y_sorted = np.sort(y)
            probs = np.linspace(0, 1, len(y_sorted), endpoint=False)[1:]
            ax.plot(x.iloc[:len(probs)], norm.ppf(probs), color="green", label="Probability")

        # Plot type
        if plot_type == "Line":
            ax.plot(x, y, color=line_color, linewidth=line_thickness, label=f'{y_col} vs {x_col}')
        elif plot_type == "Scatter":
            ax.scatter(x, y, color=scatter_color, label=f'{y_col} vs {x_col}')
        else:
            ax.plot(x, y, color=line_color, linewidth=line_thickness, label='Line')
            ax.scatter(x, y, color=scatter_color, label='Scatter')

        # Axis labels
        ax.set_xlabel(x_label, fontdict=fontdict)
        ax.set_ylabel(y_label, fontdict=fontdict)

        # Tick label format
        if tick_format == "Scientific":
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.xaxis.get_major_formatter().set_scientific(True)
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.get_major_formatter().set_scientific(True)
        elif tick_format == "Decimal":
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        # Default does nothing

        # Border thickness
        for spine in ax.spines.values():
            spine.set_linewidth(border_thickness)

        # Grid
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # Legend
        if show_legend:
            ax.legend(loc="best", fontsize=font_size)

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Plotting failed: {e}")

else:
    st.info("Please upload a CSV file to get started.")

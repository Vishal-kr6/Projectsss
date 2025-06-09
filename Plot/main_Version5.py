import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from io import BytesIO
import matplotlib.font_manager as font_manager

st.set_page_config(layout='wide', page_title="CSV Plotter")

st.title("CSV Plotter and Grapher (Smart Range Suggestions)")

csv_file = st.file_uploader("Upload your CSV file", type=["csv"])

def log10_range(data):
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data[data != 0].abs()
    if data.empty:
        return -1, 1
    return float(np.log10(data.min())), float(np.log10(data.max()))

def linear_range(data):
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return 0, 1
    return float(data.min()), float(data.max())

if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.success("CSV loaded!")
    st.dataframe(df.head())

    columns = list(df.columns)
    if len(columns) < 2:
        st.warning("CSV must have at least two columns.")
        st.stop()

    with st.sidebar:
        st.header("Plot Settings")

        x_col = st.selectbox("X column", columns, key="x_col")
        y_col = st.selectbox("Y column", columns, index=1 if len(columns) > 1 else 0, key="y_col")
        plot_type = st.selectbox("Plot Type", ["Line", "Scatter", "Line + Scatter"], key="plt_type")

        st.subheader("Axis Scales (choose separately)")
        x_scale = st.selectbox("X Axis Scale", ["Linear", "Logarithmic", "Probability"], key="x_scale")
        y_scale = st.selectbox("Y Axis Scale", ["Linear", "Logarithmic", "Probability"], key="y_scale")

        st.subheader("Axis Labels")
        x_label = st.text_input("X Axis Label", value=x_col, key="x_label")
        y_label = st.text_input("Y Axis Label", value=y_col, key="y_label")

        font_list = sorted(set(f.name for f in font_manager.fontManager.ttflist))
        font_default = font_list.index("DejaVu Sans") if "DejaVu Sans" in font_list else 0
        font_family = st.selectbox("Font Family", font_list, index=font_default, key="font_family")
        font_size = st.slider("Font size", 8, 32, 14, key="font_size")
        font_weight = st.checkbox("Bold Axis Labels", False, key="font_weight")

        st.subheader("Axis Range and Ticks")

        # X range/ticks with dynamic suggestion
        x_data = df[x_col]
        if x_scale == "Logarithmic":
            x_log_min, x_log_max = log10_range(x_data)
            st.caption(f"X log10 range: {x_log_min:.3g} to {x_log_max:.3g} (values: {10**x_log_min:.3g} to {10**x_log_max:.3g})")
            x_min_log = st.number_input("X min (log10)", value=x_log_min, key="x_min_log", format="%.4g")
            x_max_log = st.number_input("X max (log10)", value=x_log_max, key="x_max_log", format="%.4g")
            x_step_log = st.number_input("X step (log10)", value=max((x_log_max-x_log_min)/10, 0.1), min_value=1e-8, format="%.4g", key="x_step_log")
        elif x_scale == "Probability":
            st.caption("X probability range: 0.0 to 1.0")
            x_min_lin = st.number_input("X min (probability)", value=0.0, min_value=0.0, max_value=1.0, key="x_min_prob")
            x_max_lin = st.number_input("X max (probability)", value=1.0, min_value=0.0, max_value=1.0, key="x_max_prob")
            x_step_lin = st.number_input("X step (probability)", value=0.1, min_value=1e-5, max_value=1.0, format="%.4g", key="x_step_prob")
        else:
            x_lin_min, x_lin_max = linear_range(x_data)
            st.caption(f"X range: {x_lin_min:.4g} to {x_lin_max:.4g}")
            x_min_lin = st.number_input("X min", value=x_lin_min, key="x_min_lin", format="%.4g")
            x_max_lin = st.number_input("X max", value=x_lin_max, key="x_max_lin", format="%.4g")
            x_step_lin = st.number_input("X step", value=max((x_lin_max-x_lin_min)/10, 1e-6), min_value=1e-8, format="%.4g", key="x_step_lin")

        # Y range/ticks with dynamic suggestion
        y_data = df[y_col]
        if y_scale == "Logarithmic":
            y_log_min, y_log_max = log10_range(y_data)
            st.caption(f"Y log10 range: {y_log_min:.3g} to {y_log_max:.3g} (values: {10**y_log_min:.3g} to {10**y_log_max:.3g})")
            y_min_log = st.number_input("Y min (log10)", value=y_log_min, key="y_min_log", format="%.4g")
            y_max_log = st.number_input("Y max (log10)", value=y_log_max, key="y_max_log", format="%.4g")
            y_step_log = st.number_input("Y step (log10)", value=max((y_log_max-y_log_min)/10, 0.1), min_value=1e-8, format="%.4g", key="y_step_log")
        elif y_scale == "Probability":
            st.caption("Y probability range: 0.0 to 1.0")
            y_min_lin = st.number_input("Y min (probability)", value=0.0, min_value=0.0, max_value=1.0, key="y_min_prob")
            y_max_lin = st.number_input("Y max (probability)", value=1.0, min_value=0.0, max_value=1.0, key="y_max_prob")
            y_step_lin = st.number_input("Y step (probability)", value=0.1, min_value=1e-5, max_value=1.0, format="%.4g", key="y_step_prob")
        else:
            y_lin_min, y_lin_max = linear_range(y_data)
            st.caption(f"Y range: {y_lin_min:.4g} to {y_lin_max:.4g}")
            y_min_lin = st.number_input("Y min", value=y_lin_min, key="y_min_lin", format="%.4g")
            y_max_lin = st.number_input("Y max", value=y_lin_max, key="y_max_lin", format="%.4g")
            y_step_lin = st.number_input("Y step", value=max((y_lin_max-y_lin_min)/10, 1e-6), min_value=1e-8, format="%.4g", key="y_step_lin")

        st.subheader("Colors and Style")
        scatter_color = st.color_picker("Scatter color", "#1f77b4")
        line_color = st.color_picker("Line color", "#ff7f0e")
        line_thickness = st.slider("Line thickness", 1, 10, 2)
        border_thickness = st.slider("Border thickness", 1, 5, 2)
        show_legend = st.checkbox("Show Legend", True)

        st.subheader("Plot Size")
        plot_width = st.slider("Plot width (inches)", 4, 16, 8)
        plot_height = st.slider("Plot height (inches)", 3, 10, 5)

    # Main plot
    try:
        x = df[x_col].replace([np.inf, -np.inf], np.nan)
        y = df[y_col].replace([np.inf, -np.inf], np.nan)
        # Data for log axes: abs, mask zeros
        if x_scale == "Logarithmic":
            x = x.abs()
            mask_x = (x > 0)
        else:
            mask_x = pd.Series(True, index=x.index)
        if y_scale == "Logarithmic":
            y = y.abs()
            mask_y = (y > 0)
        else:
            mask_y = pd.Series(True, index=y.index)
        mask = mask_x & mask_y
        x = x[mask]
        y = y[mask]
        if len(x) == 0 or len(y) == 0:
            st.warning("No valid data to plot after processing for log/abs/zero.")
            st.stop()

        fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=120)
        fontdict = {
            "fontsize": font_size,
            "fontweight": "bold" if font_weight else "normal",
            "fontname": font_family
        }

        # X axis
        if x_scale == "Logarithmic":
            ax.set_xscale("log")
            ax.set_xlim(10**x_min_log, 10**x_max_log)
            x_ticks = 10 ** np.arange(x_min_log, x_max_log + x_step_log, x_step_log)
            ax.set_xticks(x_ticks)
        elif x_scale == "Probability":
            ax.set_xlim(x_min_lin, x_max_lin)
            ax.set_xticks(np.arange(x_min_lin, x_max_lin + x_step_lin, x_step_lin))
        else:
            ax.set_xlim(x_min_lin, x_max_lin)
            ax.set_xticks(np.arange(x_min_lin, x_max_lin + x_step_lin, x_step_lin))

        # Y axis
        if y_scale == "Logarithmic":
            ax.set_yscale("log")
            ax.set_ylim(10**y_min_log, 10**y_max_log)
            y_ticks = 10 ** np.arange(y_min_log, y_max_log + y_step_log, y_step_log)
            ax.set_yticks(y_ticks)
        elif y_scale == "Probability":
            ax.set_ylim(y_min_lin, y_max_lin)
            ax.set_yticks(np.arange(y_min_lin, y_max_lin + y_step_lin, y_step_lin))
        else:
            ax.set_ylim(y_min_lin, y_max_lin)
            ax.set_yticks(np.arange(y_min_lin, y_max_lin + y_step_lin, y_step_lin))

        # Plot
        if plot_type == "Line":
            ax.plot(x, y, color=line_color, linewidth=line_thickness, label=f'{y_col} vs {x_col}')
        elif plot_type == "Scatter":
            ax.scatter(x, y, color=scatter_color, s=10, label=f'{y_col} vs {x_col}')
        else:
            ax.plot(x, y, color=line_color, linewidth=line_thickness, label='Line')
            ax.scatter(x, y, color=scatter_color, s=10, label='Scatter')

        # Labels and style
        ax.set_xlabel(x_label, fontdict=fontdict)
        ax.set_ylabel(y_label, fontdict=fontdict)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.margins(0.03)
        for spine in ax.spines.values():
            spine.set_linewidth(border_thickness)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        if show_legend:
            ax.legend(loc="best", fontsize=font_size)
        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
        st.pyplot(fig, use_container_width=True)
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png", bbox_inches="tight")
        st.download_button(
            label="Download Plot as PNG",
            data=img_buf.getvalue(),
            file_name="plot.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Plotting failed: {e}")

else:
    st.info("Please upload a CSV file to get started.")

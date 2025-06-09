import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter
from io import BytesIO
import matplotlib.font_manager as font_manager

st.set_page_config(layout='wide', page_title="CSV Plotter")

st.title("CSV Plotter and Grapher (with Log Absolute & Download)")

csv_file = st.file_uploader("Upload your CSV file", type=["csv"])

if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.success("CSV loaded!")
    st.dataframe(df.head())

    columns = list(df.columns)
    if len(columns) < 2:
        st.warning("CSV must have at least two columns.")
        st.stop()

    # Sidebar for settings
    with st.sidebar:
        st.header("Plot Settings")

        x_col = st.selectbox("X column", columns)
        y_col = st.selectbox("Y column", columns, index=1 if len(columns) > 1 else 0)
        plot_type = st.selectbox("Plot Type", ["Line", "Scatter", "Line + Scatter"])

        st.markdown("---")
        st.subheader("Axis Scales (choose separately)")
        x_scale = st.selectbox("X Axis Scale", ["Linear", "Logarithmic"])
        y_scale = st.selectbox("Y Axis Scale", ["Linear", "Logarithmic"])

        st.markdown("---")
        st.subheader("Axis Labels")
        x_label = st.text_input("X Axis Label", value=x_col)
        y_label = st.text_input("Y Axis Label", value=y_col)

        font_list = sorted(set(f.name for f in font_manager.fontManager.ttflist))
        font_default = font_list.index("DejaVu Sans") if "DejaVu Sans" in font_list else 0
        font_family = st.selectbox("Font Family", font_list, index=font_default)
        font_size = st.slider("Font size", 8, 32, 14)
        font_weight = st.checkbox("Bold Axis Labels", False)

        st.markdown("---")
        st.subheader("Axis Range and Step")

        # Prepare X data, with abs for log if needed
        x_data = df[x_col].replace([np.inf, -np.inf], np.nan).dropna()
        if x_scale == "Logarithmic":
            x_data = x_data[x_data != 0].dropna()
            x_data = x_data.abs()
            if x_data.empty:
                x_abs_min, x_abs_max = 1e-6, 1
                st.warning("No nonzero data for X axis; using dummy log scale.")
            else:
                x_abs_min, x_abs_max = float(x_data.min()), float(x_data.max())
            st.caption(f"X abs range (for log): {x_abs_min:.4g} to {x_abs_max:.4g}")
            xm = st.number_input("X min (abs, log scale)", value=x_abs_min, key="x_min")
            xM = st.number_input("X max (abs, log scale)", value=x_abs_max, key="x_max")
            xs = st.number_input("X step (log units)", value=max((np.log10(x_abs_max)-np.log10(x_abs_min))/10, 0.1), min_value=1e-8, format="%.4g", key="x_step")
        else:
            x_lin_min, x_lin_max = float(x_data.min()), float(x_data.max())
            st.caption(f"X range: {x_lin_min:.4g} to {x_lin_max:.4g}")
            xm = st.number_input("X min", value=x_lin_min, key="x_min")
            xM = st.number_input("X max", value=x_lin_max, key="x_max")
            xs = st.number_input("X step", value=max((x_lin_max-x_lin_min)/10, 1e-6), min_value=1e-8, format="%.4g", key="x_step")

        # Prepare Y data, with abs for log if needed
        y_data = df[y_col].replace([np.inf, -np.inf], np.nan).dropna()
        if y_scale == "Logarithmic":
            y_data = y_data[y_data != 0].dropna()
            y_data = y_data.abs()
            if y_data.empty:
                y_abs_min, y_abs_max = 1e-6, 1
                st.warning("No nonzero data for Y axis; using dummy log scale.")
            else:
                y_abs_min, y_abs_max = float(y_data.min()), float(y_data.max())
            st.caption(f"Y abs range (for log): {y_abs_min:.4g} to {y_abs_max:.4g}")
            ym = st.number_input("Y min (abs, log scale)", value=y_abs_min, key="y_min")
            yM = st.number_input("Y max (abs, log scale)", value=y_abs_max, key="y_max")
            ys = st.number_input("Y step (log units)", value=max((np.log10(y_abs_max)-np.log10(y_abs_min))/10, 0.1), min_value=1e-8, format="%.4g", key="y_step")
        else:
            y_lin_min, y_lin_max = float(y_data.min()), float(y_data.max())
            st.caption(f"Y range: {y_lin_min:.4g} to {y_lin_max:.4g}")
            ym = st.number_input("Y min", value=y_lin_min, key="y_min")
            yM = st.number_input("Y max", value=y_lin_max, key="y_max")
            ys = st.number_input("Y step", value=max((y_lin_max-y_lin_min)/10, 1e-6), min_value=1e-8, format="%.4g", key="y_step")

        st.markdown("---")
        st.subheader("Colors and Style")
        scatter_color = st.color_picker("Scatter color", "#1f77b4")
        line_color = st.color_picker("Line color", "#ff7f0e")
        line_thickness = st.slider("Line thickness", 1, 10, 2)
        border_thickness = st.slider("Border thickness", 1, 5, 2)
        show_legend = st.checkbox("Show Legend", True)

        st.markdown("---")
        st.subheader("Plot Size")
        plot_width = st.slider("Plot width (inches)", 4, 16, 8)
        plot_height = st.slider("Plot height (inches)", 3, 10, 5)

    # Main area for live plot
    try:
        # Prepare data for plotting
        x = df[x_col].replace([np.inf, -np.inf], np.nan)
        y = df[y_col].replace([np.inf, -np.inf], np.nan)
        # For log axes, use abs and mask zeros
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

        # Axis scale and ticks
        if x_scale == "Logarithmic":
            ax.set_xscale("log")
            x_ticks = np.logspace(np.log10(xm), np.log10(xM), int((np.log10(xM)-np.log10(xm))/xs)+1)
            ax.set_xlim(xm, xM)
            ax.set_xticks(x_ticks)
        else:
            ax.set_xlim(xm, xM)
            ax.set_xticks(np.arange(xm, xM+xs, xs))

        if y_scale == "Logarithmic":
            ax.set_yscale("log")
            y_ticks = np.logspace(np.log10(ym), np.log10(yM), int((np.log10(yM)-np.log10(ym))/ys)+1)
            ax.set_ylim(ym, yM)
            ax.set_yticks(y_ticks)
        else:
            ax.set_ylim(ym, yM)
            ax.set_yticks(np.arange(ym, yM+ys, ys))

        # Plot type
        if plot_type == "Line":
            ax.plot(x, y, color=line_color, linewidth=line_thickness, label=f'{y_col} vs {x_col}')
        elif plot_type == "Scatter":
            ax.scatter(x, y, color=scatter_color, s=10, label=f'{y_col} vs {x_col}')
        else:
            ax.plot(x, y, color=line_color, linewidth=line_thickness, label='Line')
            ax.scatter(x, y, color=scatter_color, s=10, label='Scatter')

        # Axis labels
        ax.set_xlabel(x_label, fontdict=fontdict)
        ax.set_ylabel(y_label, fontdict=fontdict)

        # Format ticks (always clear, not cut by borders)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.margins(0.03)

        # Border
        for spine in ax.spines.values():
            spine.set_linewidth(border_thickness)

        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        if show_legend:
            ax.legend(loc="best", fontsize=font_size)

        plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])

        st.pyplot(fig, use_container_width=True)

        # Download button
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

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FuncFormatter
from io import BytesIO
import matplotlib.font_manager as font_manager
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout='wide', page_title="CSV Plotter")

st.title("CSV Plotter and Grapher: Multi-Fit with R² for Groups")

csv_file = st.file_uploader("Upload your CSV file", type=["csv"])

def suggest_real_range(data):
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data_nonzero = data[data != 0].abs()
    if not data_nonzero.empty:
        return float(data_nonzero.min()), float(data_nonzero.max())
    elif not data.empty:
        return float(abs(data.min())), float(abs(data.max()))
    else:
        return 0.001, 10.0

def suggest_linear_range(data):
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty:
        return 0.0, 1.0
    return float(data.min()), float(data.max())

if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.success("CSV loaded!")
    st.dataframe(df.head())

    columns = list(df.columns)
    if len(columns) < 3:
        st.warning("CSV must have at least three columns: X, Y, and Group (e.g. temperature).")
        st.stop()

    with st.sidebar:
        st.header("Plot Settings")

        x_col = st.selectbox("X column", columns, key="x_col")
        y_col = st.selectbox("Y column", columns, index=1 if len(columns) > 1 else 0, key="y_col")
        group_col = st.selectbox("Group column (e.g. temperature)", [c for c in columns if c not in [x_col, y_col]], key="group_col")

        st.subheader("Axis Scales (choose separately)")
        x_scale = st.selectbox("X Axis Scale", ["Linear", "Logarithmic"], key="x_scale")
        y_scale = st.selectbox("Y Axis Scale", ["Linear", "Logarithmic"], key="y_scale")

        st.subheader("Number Format")
        x_tick_format = st.selectbox("X Axis Number Format", ["Default", "Scientific", "Decimal"], key="x_tick_fmt")
        y_tick_format = st.selectbox("Y Axis Number Format", ["Default", "Scientific", "Decimal"], key="y_tick_fmt")

        st.subheader("Axis Labels")
        x_label = st.text_input("X Axis Label", value=x_col, key="x_label")
        y_label = st.text_input("Y Axis Label", value=y_col, key="y_label")

        font_list = sorted(set(f.name for f in font_manager.fontManager.ttflist))
        font_default = font_list.index("DejaVu Sans") if "DejaVu Sans" in font_list else 0
        font_family = st.selectbox("Font Family", font_list, index=font_default, key="font_family")
        font_size = st.slider("Font size", 8, 32, 14, key="font_size")
        font_weight = st.checkbox("Bold Axis Labels", False, key="font_weight")

        st.subheader("Axis Range and Ticks")

        # X range/ticks
        x_data = df[x_col]
        if x_scale == "Logarithmic":
            x_min_suggest, x_max_suggest = suggest_real_range(x_data)
            st.caption(f"X axis abs(nonzero) range: {x_min_suggest:.4g} to {x_max_suggest:.4g}")
            x_min = st.number_input("X min (log scale, real value)", value=x_min_suggest, min_value=1e-12, format="%.4g", key="x_min")
            x_max = st.number_input("X max (log scale, real value)", value=x_max_suggest, min_value=x_min+1e-12, format="%.4g", key="x_max")
            x_ticks_count = st.number_input("Number of X ticks", min_value=2, max_value=50, value=10, step=1, key="x_ticks_count")
        else:
            x_lin_min, x_lin_max = suggest_linear_range(x_data)
            st.caption(f"X range: {x_lin_min:.4g} to {x_lin_max:.4g}")
            x_min = st.number_input("X min", value=x_lin_min, key="x_min_lin", format="%.4g")
            x_max = st.number_input("X max", value=x_lin_max, min_value=x_min+1e-12, key="x_max_lin", format="%.4g")
            x_step = st.number_input("X step", value=max((x_lin_max-x_lin_min)/10, 1e-6), min_value=1e-8, format="%.4g", key="x_step_lin")

        # Y range/ticks
        y_data = df[y_col]
        if y_scale == "Logarithmic":
            y_min_suggest, y_max_suggest = suggest_real_range(y_data)
            st.caption(f"Y axis abs(nonzero) range: {y_min_suggest:.4g} to {y_max_suggest:.4g}")
            y_min = st.number_input("Y min (log scale, real value)", value=y_min_suggest, min_value=1e-12, format="%.4g", key="y_min")
            y_max = st.number_input("Y max (log scale, real value)", value=y_max_suggest, min_value=y_min+1e-12, format="%.4g", key="y_max")
            y_ticks_count = st.number_input("Number of Y ticks", min_value=2, max_value=50, value=10, step=1, key="y_ticks_count")
        else:
            y_lin_min, y_lin_max = suggest_linear_range(y_data)
            st.caption(f"Y range: {y_lin_min:.4g} to {y_lin_max:.4g}")
            y_min = st.number_input("Y min", value=y_lin_min, key="y_min_lin", format="%.4g")
            y_max = st.number_input("Y max", value=y_lin_max, min_value=y_min+1e-12, key="y_max_lin", format="%.4g")
            y_step = st.number_input("Y step", value=max((y_lin_max-y_lin_min)/10, 1e-6), min_value=1e-8, format="%.4g", key="y_step_lin")

        st.subheader("Colors and Style")
        group_markers = ['o', '^', 's', 'D', 'X', 'v', 'P', '*', '<', '>', 'H']
        group_colors = ['black', 'red', 'blue', 'green', 'magenta', 'orange', 'cyan', 'purple', 'brown', 'grey']
        fit_colors = ['black', 'red', 'blue', 'green', 'magenta', 'orange', 'cyan', 'purple', 'brown', 'grey']

        border_thickness = st.slider("Border thickness", 1, 5, 2)
        show_legend = st.checkbox("Show Legend", True)

        st.subheader("Plot Size")
        plot_width = st.slider("Plot width (inches)", 4, 16, 8)
        plot_height = st.slider("Plot height (inches)", 3, 10, 5)

    # Main plot
    try:
        group_vals = df[group_col].dropna().unique()
        group_vals = sorted(group_vals)

        def mask_log(x, y):
            mask = pd.Series([True]*len(x))
            if x_scale == "Logarithmic":
                mask = mask & (x.abs() > 0)
            if y_scale == "Logarithmic":
                mask = mask & (y.abs() > 0)
            return mask

        fig, ax = plt.subplots(figsize=(plot_width, plot_height), dpi=120)
        fontdict = {
            "fontsize": font_size,
            "fontweight": "bold" if font_weight else "normal",
            "fontname": font_family
        }

        # X axis
        if x_scale == "Logarithmic":
            ax.set_xscale("log")
            ax.set_xlim(x_min, x_max)
            x_ticks = np.logspace(np.log10(x_min), np.log10(x_max), int(x_ticks_count))
            ax.set_xticks(x_ticks)
            if x_tick_format == "Scientific":
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.xaxis.get_major_formatter().set_scientific(True)
                ax.xaxis.get_major_formatter().set_powerlimits((-2, 2))
            elif x_tick_format == "Decimal":
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(np.arange(x_min, x_max + x_step, x_step))
            if x_tick_format == "Scientific":
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.xaxis.get_major_formatter().set_scientific(True)
                ax.xaxis.get_major_formatter().set_powerlimits((-2, 2))
            elif x_tick_format == "Decimal":
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))

        # Y axis
        if y_scale == "Logarithmic":
            ax.set_yscale("log")
            ax.set_ylim(y_min, y_max)
            y_ticks = np.logspace(np.log10(y_min), np.log10(y_max), int(y_ticks_count))
            ax.set_yticks(y_ticks)
            if y_tick_format == "Scientific":
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.yaxis.get_major_formatter().set_scientific(True)
                ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))
            elif y_tick_format == "Decimal":
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_ylim(y_min, y_max)
            ax.set_yticks(np.arange(y_min, y_max + y_step, y_step))
            if y_tick_format == "Scientific":
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.yaxis.get_major_formatter().set_scientific(True)
                ax.yaxis.get_major_formatter().set_powerlimits((-2, 2))
            elif y_tick_format == "Decimal":
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

        # Plot points and fit for each group
        for i, group in enumerate(group_vals):
            dfg = df[df[group_col] == group]
            # Drop rows with NA in x or y
            dfg = dfg[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            if dfg.empty:
                continue
            xg = dfg[x_col]
            yg = dfg[y_col]
            if x_scale == "Logarithmic":
                xg = xg.abs()
            if y_scale == "Logarithmic":
                yg = yg.abs()
            # After abs, mask for >0 if log
            mask = mask_log(xg, yg)
            xg = xg[mask].reset_index(drop=True)
            yg = yg[mask].reset_index(drop=True)
            if len(xg) == 0 or len(yg) == 0:
                continue
            marker = group_markers[i % len(group_markers)]
            color = group_colors[i % len(group_colors)]
            fit_color = fit_colors[i % len(fit_colors)]
            # Scatter points
            ax.scatter(xg, yg, color=color, marker=marker, s=40, label=f"{group}")
            # Fit line
            X_fit = np.log10(xg).values.reshape(-1, 1) if x_scale == "Logarithmic" else xg.values.reshape(-1, 1)
            y_fit = np.log10(yg).values if y_scale == "Logarithmic" else yg.values
            reg = LinearRegression().fit(X_fit, y_fit)
            # Create fit line in original scale
            x_fit_range = np.linspace(xg.min(), xg.max(), 100)
            if x_scale == "Logarithmic":
                X_plot = np.log10(x_fit_range).reshape(-1, 1)
            else:
                X_plot = x_fit_range.reshape(-1, 1)
            y_pred = reg.predict(X_plot)
            # R2 value
            y_pred_data = reg.predict(X_fit)
            r2 = r2_score(y_fit, y_pred_data)
            if y_scale == "Logarithmic":
                y_plot = np.power(10, y_pred)
            else:
                y_plot = y_pred
            ax.plot(x_fit_range, y_plot, label=f"Linear Fit {group}", color=fit_color, linewidth=2)
            # R2 annotation: place near the data
            r2_x = xg.mean()
            r2_y = yg.max()
            ax.text(
                r2_x, r2_y, f"$R^2$={r2:.5f}",
                color=fit_color, fontsize=font_size, fontweight="bold",
                fontname=font_family
            )

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

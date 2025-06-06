import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

st.set_page_config(layout='wide', page_title="CSV Plotter")

st.title("CSV Plotter and Grapher (Streamlit Edition)")

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

    with st.form("plot_form"):
        st.subheader("Plot Settings")

        col1, col2 = st.columns(2)
        x_col = col1.selectbox("X column", columns)
        y_col = col2.selectbox("Y column", columns, index=1 if len(columns) > 1 else 0)

        plot_type = st.selectbox("Plot Type", ["Line", "Scatter", "Line + Scatter"])

        x_scale = st.selectbox("X Axis Scale", ["Linear", "Logarithmic (log10)", "Scientific", "Decimal"])
        y_scale = st.selectbox("Y Axis Scale", ["Linear", "Logarithmic (log10)", "Scientific", "Decimal"])

        st.markdown("#### Axis Labels and Style")
        label_font_size = st.slider("Font size", 8, 32, 14)
        label_font_weight = st.checkbox("Bold Axis Labels", False)
        label_font = {"fontsize": label_font_size, "fontweight": "bold" if label_font_weight else "normal"}

        x_label = st.text_input("X Axis Label", x_col)
        y_label = st.text_input("Y Axis Label", y_col)

        st.markdown("#### Axis Range and Step")
        x_min, x_max = float(df[x_col].min()), float(df[x_col].max())
        y_min, y_max = float(df[y_col].min()), float(df[y_col].max())
        xm = st.number_input("X min", value=x_min, key="x_min")
        xM = st.number_input("X max", value=x_max, key="x_max")
        xs = st.number_input("X step", value=max((x_max-x_min)/10, 1e-6), min_value=1e-6, format="%.6f", key="x_step")
        ym = st.number_input("Y min", value=y_min, key="y_min")
        yM = st.number_input("Y max", value=y_max, key="y_max")
        ys = st.number_input("Y step", value=max((y_max-y_min)/10, 1e-6), min_value=1e-6, format="%.6f", key="y_step")

        st.markdown("#### Colors and Style")
        scatter_color = st.color_picker("Scatter color", "#1f77b4")
        line_color = st.color_picker("Line color", "#ff7f0e")
        line_thickness = st.slider("Line thickness", 1, 10, 2)
        border_thickness = st.slider("Border thickness", 1, 10, 2)
        show_legend = st.checkbox("Show Legend", True)

        submitted = st.form_submit_button("Plot!")

    if submitted:
        fig, ax = plt.subplots(figsize=(8, 5))
        try:
            x = df[x_col]
            y = df[y_col]

            # Axis scale
            if x_scale.startswith("Log"):
                ax.set_xscale("log")
            if y_scale.startswith("Log"):
                ax.set_yscale("log")

            # Plot
            if plot_type == "Line":
                ax.plot(x, y, color=line_color, linewidth=line_thickness, label=f'{y_col} vs {x_col}')
            elif plot_type == "Scatter":
                ax.scatter(x, y, color=scatter_color, label=f'{y_col} vs {x_col}')
            else:
                ax.plot(x, y, color=line_color, linewidth=line_thickness, label='Line')
                ax.scatter(x, y, color=scatter_color, label='Scatter')

            # Axis labels
            ax.set_xlabel(x_label, fontdict=label_font)
            ax.set_ylabel(y_label, fontdict=label_font)

            # Ticks
            ax.set_xticks(np.arange(xm, xM+xs, xs))
            ax.set_yticks(np.arange(ym, yM+ys, ys))

            # Axis formatting
            if x_scale == "Scientific":
                ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
            if y_scale == "Scientific":
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
            if x_scale == "Decimal":
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if y_scale == "Decimal":
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # Border thickness
            for spine in ax.spines.values():
                spine.set_linewidth(border_thickness)

            # Grid
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

            # Legend
            if show_legend:
                ax.legend(loc="best", fontsize=label_font_size)

            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Plotting failed: {e}")

else:
    st.info("Please upload a CSV file to get started.")
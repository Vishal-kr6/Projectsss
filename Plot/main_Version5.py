import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from io import BytesIO
import matplotlib.font_manager as font_manager
from conversion_utils import convert_units, plot_folder_of_csvs

st.set_page_config(layout='wide', page_title="CSV Plotter")

st.title("CSV Plotter and Grapher (with conversions and folder support)")

tab1, tab2 = st.tabs(["Single CSV Upload", "Batch Folder Plot"])

with tab1:
    csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if csv_file is not None:
        df = pd.read_csv(csv_file)
        df = convert_units(df)
        st.success("CSV loaded with conversions!")
        st.dataframe(df.head())
        columns = list(df.columns)
        if len(columns) < 2:
            st.warning("CSV must have at least two columns.")
            st.stop()

        with st.sidebar:
            st.header("Plot Settings (Single File)")
            # All your sidebar settings...
            x_col = st.selectbox("X column", columns, key="x_col")
            y_col = st.selectbox("Y column", columns, index=1 if len(columns) > 1 else 0, key="y_col")
            plot_type = st.selectbox("Plot Type", ["Line", "Scatter", "Line + Scatter"], key="plt_type")

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
            bold_lines = st.checkbox("Bold Plot Lines", False, key="bold_lines")

            st.subheader("Axis Range and Ticks")
            x_data = df[x_col]
            y_data = df[y_col]
            x_min, x_max, x_step = float(x_data.min()), float(x_data.max()), max((x_data.max()-x_data.min())/10, 1e-6)
            y_min, y_max, y_step = float(y_data.min()), float(y_data.max()), max((y_data.max()-y_data.min())/10, 1e-6)

            st.subheader("Colors and Style")
            scatter_color = st.color_picker("Scatter color", "#1f77b4")
            line_color = st.color_picker("Line color", "#ff7f0e")
            line_thickness = st.slider("Line thickness", 1, 10, 2)
            border_thickness = st.slider("Border thickness", 1, 5, 2)
            show_legend = st.checkbox("Show Legend", True)

            st.subheader("Plot Size")
            plot_width = st.slider("Plot width (inches)", 4, 16, 8)
            plot_height = st.slider("Plot height (inches)", 3, 10, 5)

        # Plotting function (refactored so it's reusable below)
        def make_plot(df, fname, **kwargs):
            x = df[kwargs['x_col']]
            y = df[kwargs['y_col']]
            fig, ax = plt.subplots(figsize=(kwargs['plot_width'], kwargs['plot_height']), dpi=120)
            fontdict = {
                "fontsize": kwargs['font_size'],
                "fontweight": "bold" if kwargs['font_weight'] else "normal",
                "fontname": kwargs['font_family']
            }
            lw = 3 if kwargs['bold_lines'] else kwargs['line_thickness']

            # Plot logic (simplified, see your full logic for complete support)
            if kwargs['plot_type'] == "Line":
                ax.plot(x, y, color=kwargs['line_color'], linewidth=lw, label=f"{kwargs['y_col']} vs {kwargs['x_col']}")
            elif kwargs['plot_type'] == "Scatter":
                ax.scatter(x, y, color=kwargs['scatter_color'], s=10, label=f"{kwargs['y_col']} vs {kwargs['x_col']}")
            else:
                ax.plot(x, y, color=kwargs['line_color'], linewidth=lw, label='Line')
                ax.scatter(x, y, color=kwargs['scatter_color'], s=10, label='Scatter')

            ax.set_xlabel(kwargs['x_label'], fontdict=fontdict)
            ax.set_ylabel(kwargs['y_label'], fontdict=fontdict)
            for spine in ax.spines.values():
                spine.set_linewidth(kwargs['border_thickness'])
            if kwargs['show_legend']:
                ax.legend(loc="best", fontsize=kwargs['font_size'])
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
            return fig

        # Main plot
        try:
            fig = make_plot(
                df, "main",
                x_col=x_col, y_col=y_col,
                plot_type=plot_type,
                x_label=x_label, y_label=y_label,
                font_family=font_family, font_size=font_size,
                font_weight=font_weight, bold_lines=bold_lines,
                scatter_color=scatter_color, line_color=line_color,
                line_thickness=line_thickness, border_thickness=border_thickness,
                plot_width=plot_width, plot_height=plot_height,
                show_legend=show_legend
            )
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

with tab2:
    folder_path = st.text_input("Enter path to folder with CSV files (on server):")
    with st.sidebar:
        st.header("Plot Settings (Batch)")
        # You can copy sidebar controls from above or make them simpler
        x_col = st.text_input("X column for all files", value="", key="x_col2")
        y_col = st.text_input("Y column for all files", value="", key="y_col2")
        plot_type = st.selectbox("Plot Type", ["Line", "Scatter", "Line + Scatter"], key="plt_type2")
        font_weight = st.checkbox("Bold Axis Labels", False, key="font_weight2")
        bold_lines = st.checkbox("Bold Plot Lines", False, key="bold_lines2")
        scatter_color = st.color_picker("Scatter color", "#1f77b4", key="scatter_color2")
        line_color = st.color_picker("Line color", "#ff7f0e", key="line_color2")
        line_thickness = st.slider("Line thickness", 1, 10, 2, key="line_thickness2")
        border_thickness = st.slider("Border thickness", 1, 5, 2, key="border_thickness2")
        font_list = sorted(set(f.name for f in font_manager.fontManager.ttflist))
        font_default = font_list.index("DejaVu Sans") if "DejaVu Sans" in font_list else 0
        font_family = st.selectbox("Font Family", font_list, index=font_default, key="font_family2")
        font_size = st.slider("Font size", 8, 32, 14, key="font_size2")
        plot_width = st.slider("Plot width (inches)", 4, 16, 8, key="plot_width2")
        plot_height = st.slider("Plot height (inches)", 3, 10, 5, key="plot_height2")
        show_legend = st.checkbox("Show Legend", True, key="show_legend2")

    if folder_path and x_col and y_col:
        def batch_make_plot(df, fname, **kwargs):
            x = df[kwargs['x_col']]
            y = df[kwargs['y_col']]
            fig, ax = plt.subplots(figsize=(kwargs['plot_width'], kwargs['plot_height']), dpi=120)
            fontdict = {
                "fontsize": kwargs['font_size'],
                "fontweight": "bold" if kwargs['font_weight'] else "normal",
                "fontname": kwargs['font_family']
            }
            lw = 3 if kwargs['bold_lines'] else kwargs['line_thickness']
            if kwargs['plot_type'] == "Line":
                ax.plot(x, y, color=kwargs['line_color'], linewidth=lw, label=f"{kwargs['y_col']} vs {kwargs['x_col']}")
            elif kwargs['plot_type'] == "Scatter":
                ax.scatter(x, y, color=kwargs['scatter_color'], s=10, label=f"{kwargs['y_col']} vs {kwargs['x_col']}")
            else:
                ax.plot(x, y, color=kwargs['line_color'], linewidth=lw, label='Line')
                ax.scatter(x, y, color=kwargs['scatter_color'], s=10, label='Scatter')
            ax.set_xlabel(kwargs['x_col'], fontdict=fontdict)
            ax.set_ylabel(kwargs['y_col'], fontdict=fontdict)
            for spine in ax.spines.values():
                spine.set_linewidth(kwargs['border_thickness'])
            if kwargs['show_legend']:
                ax.legend(loc="best", fontsize=kwargs['font_size'])
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
            return fig

        try:
            figs = plot_folder_of_csvs(
                folder_path,
                batch_make_plot,
                x_col=x_col, y_col=y_col,
                plot_type=plot_type, font_weight=font_weight, bold_lines=bold_lines,
                scatter_color=scatter_color, line_color=line_color,
                line_thickness=line_thickness, border_thickness=border_thickness,
                font_family=font_family, font_size=font_size,
                plot_width=plot_width, plot_height=plot_height,
                show_legend=show_legend
            )
            for fname, fig in figs:
                st.subheader(f"Plot for: {fname}")
                if isinstance(fig, str):
                    st.error(fig)
                else:
                    st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Batch plotting failed: {e}")
    else:
        st.info("Enter a valid folder path and column names for batch plotting.")

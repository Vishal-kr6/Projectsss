import os
import pandas as pd
import numpy as np

def convert_units(df):
    if 'Ampiere' in df.columns:
        df['Ampiere_mA'] = df['Ampiere'] * 1e3
        df['Ampiere_uA'] = df['Ampiere'] * 1e6
        df['Ampiere_nA'] = df['Ampiere'] * 1e9
        df['log_Ampiere'] = np.log(np.abs(df['Ampiere'].replace(0, np.nan)))
    if 'Voltage' in df.columns:
        df['Voltage_mV'] = df['Voltage'] * 1e3
        df['Voltage_uV'] = df['Voltage'] * 1e6
        df['Voltage_nV'] = df['Voltage'] * 1e9
        df['log_Voltage'] = np.log(np.abs(df['Voltage'].replace(0, np.nan)))
    return df

def plot_folder_of_csvs(folder_path, plot_func, **kwargs):
    """
    Applies plot_func to all CSV files in the folder.
    plot_func should take (df, filename, **kwargs) and return a matplotlib figure.
    Returns: list of (filename, fig)
    """
    figs = []
    if not os.path.isdir(folder_path):
        raise ValueError("Provided folder path does not exist.")
    for csv in os.listdir(folder_path):
        if csv.lower().endswith('.csv'):
            path = os.path.join(folder_path, csv)
            try:
                df = pd.read_csv(path)
                df = convert_units(df)
                fig = plot_func(df, csv, **kwargs)
                figs.append((csv, fig))
            except Exception as e:
                figs.append((csv, f"Error: {repr(e)}"))
    return figs

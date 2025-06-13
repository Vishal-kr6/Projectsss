import os
import pandas as pd
import numpy as np

def convert_units(df):
    # Current conversions (Ampiere)
    if 'Ampiere' in df.columns:
        df['Ampiere_mA'] = df['Ampiere'] * 1e3
        df['Ampiere_uA'] = df['Ampiere'] * 1e6
        df['Ampiere_nA'] = df['Ampiere'] * 1e9
        df['log_Ampiere'] = np.log(np.abs(df['Ampiere'].replace(0, np.nan)))
    # Voltage conversions (Voltage)
    if 'Voltage' in df.columns:
        df['Voltage_mV'] = df['Voltage'] * 1e3
        df['Voltage_uV'] = df['Voltage'] * 1e6
        df['Voltage_nV'] = df['Voltage'] * 1e9
        df['log_Voltage'] = np.log(np.abs(df['Voltage'].replace(0, np.nan)))
    return df

def list_csv_files(folder_path):
    """List all CSV files in a directory."""
    return [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

def batch_load_and_convert(folder_path):
    """Yield (filename, DataFrame with conversions) for each CSV in folder."""
    for fname in list_csv_files(folder_path):
        csv_path = os.path.join(folder_path, fname)
        df = pd.read_csv(csv_path)
        df = convert_units(df)
        yield fname, df

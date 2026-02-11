import pandas as pd
import numpy as np
import pdb

def read_isd_csv(path):
    """
    Reads weather data from the Integrated Surface Dataset (ISD).
    
    Parameters:
    path (str): Path to the CSV file containing weather data.

    Returns:
    pd.DataFrame: A DataFrame with processed weather data containing the following columns:
    TMP: air temperature (degrees Celsius)
    DEW: dewpoint temperature (degrees Celsius)
    WND: wind speed (m/s)
    SLP: sea level pressure (Pa)
    """

    try:
        # Load input data
        df_input = pd.read_csv(path, delimiter=',', dtype=str)
    except FileNotFoundError:
        raise FileNotFoundError("The specified file was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")

    # Process dates
    try:
        dates = pd.to_datetime(df_input['DATE'].values.astype(str)).round('1h')
    except Exception as e:
        raise ValueError(f"Error processing dates: {e}")

    # Initialize output dataframe
    df_output = pd.DataFrame(index=dates)

    # Function to process temperature and dew-point data
    def process_temp_data(column_name):
        ts = df_input[column_name].str.split(pat=',', expand=True)
        ts_vals = ts.iloc[:, 0].astype(float).values / 10  # Convert to degrees
        ts_qc = ts.iloc[:, 1].values
        ts_vals[(ts_qc != '1') & (ts_qc != '5')] = np.nan  # Apply QC
        return ts_vals

    # Process temperature and dew-point data
    df_output['TMP'] = process_temp_data('TMP')
    df_output['DEW'] = process_temp_data('DEW')

    # Wind speed processing
    if 'WND' in df_input.columns:
        ts = df_input['WND'].str.split(pat=',', expand=True)
        ts_vals = ts.iloc[:, 3].astype(float).values / 10
        ts_vals[ts_vals > 500] = np.nan  # Filter unrealistic values
        ts_qc = ts.iloc[:, 1].values
        ts_vals[(ts_qc != '1') & (ts_qc != '5')] = np.nan        
        df_output['WND'] = ts_vals
    
    '''
    # Surface pressure processing
    if ('SLP' in df_input.columns) & ('TMP' in df_input.columns):
        ts = df_input['SLP'].str.split(pat=',', expand=True)
        ts_vals = ts.iloc[:, 0].astype(float).values * 10  # Convert to Pa
        ts_qc = ts.iloc[:, 1].values
        ts_vals[(ts_qc != '1') & (ts_qc != '5')] = np.nan

        # Temperature correction
        Elevation = float(df_input['ELEVATION'].values[0])
        c_dt_dh = -0.006
        c_m = 0.02896
        c_r = 8.314
        c_g = 9.807
        Temp = df_output['TMP'] + 273.15  # Convert to Kelvin
        ts_vals = ts_vals * (1.0 + (c_dt_dh * Elevation) / (Temp - c_dt_dh * Elevation)) ** (c_m * c_g / c_r / (-c_dt_dh))
        df_output['SLP'] = ts_vals
    '''
    
    return df_output
    
def dewpoint_to_rh(Tdew,Temp):
    """
    Convert dewpoint temperature to relative humidity.

    Parameters:
    - Tdew (float): Dewpoint temperature in degrees Celsius.
    - Temp (float): Ambient air temperature in degrees Celsius.

    Returns:
    - float: Relative humidity in percentage.

    Source: https://www.hatchability.com/Vaisala.pdf (Equation 12)
    """
    m = 7.59138
    Tn = 240.7263
    return 100 * 10**(m * ((Tdew / (Tdew + Tn)) - (Temp / (Temp + Tn))))

def gen_heat_index(temp, rh):
    """
    Calculate the National Weather Service (NWS) heat index.

    Parameters:
    - temp (float): Ambient air temperature in degrees Celsius.
    - rh (float): Relative humidity in %.

    Returns:
    - float: Heat index in degrees Celsius.

    Source: https://en.wikipedia.org/wiki/Heat_index#Formula
    """
    
    # Constants for the heat index calculation
    c1 = -8.78469475556
    c2 = 1.61139411
    c3 = 2.33854883889
    c4 = -0.14611605
    c5 = -0.012308094
    c6 = -0.0164248277778
    c7 = 0.002211732
    c8 = 0.00072546
    c9 = -0.000003582
    return c1 + c2*temp + c3*rh + c4*temp*rh + c5*temp*temp + c6*rh*rh + c7*temp*temp*rh + c8*temp*rh*rh + c9*temp*temp*rh*rh
    
def hargreaves_samani_1982(tmin, tmax, tmean, lat, doy):
    """
    Calculate potential evaporation using the Hargreaves and Samani (1982) method.

    Inputs:
    - tmin: Array of daily minimum temperatures in degrees Celsius.
    - tmax: Array of daily maximum temperatures in degrees Celsius.
    - tmean: Array of daily mean temperatures in degrees Celsius.
    - lat: Latitude in degrees.
    - doy: Array of day-of-year corresponding to temperature data.

    Output:
    - pe: Array of potential evaporation values in mm/day.
    """
    
    latitude = np.deg2rad(lat)  # Convert latitude to radians
        
    pe = np.zeros_like(tmin, dtype=np.float32)  # Initialize potential evaporation array with NaN values
    SOLAR_CONSTANT = 0.0820  # Solar constant
    
    for ii in range(len(pe)):
        trange = np.maximum(0, tmax[ii] - tmin[ii])  # Ensure trange is non-negative
        
        # Calculate solar declination
        sol_dec = 0.409 * np.sin(((2.0 * np.pi / 365.0) * doy[ii] - 1.39))
        
        # Calculate sunset hour angle
        sha = np.arccos(np.clip(-np.tan(latitude) * np.tan(sol_dec), -1, 1))
    
        # Calculate inverse relative distance Earth-Sun
        ird = 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * doy[ii]))
        
        # Calculate extraterrestrial radiation
        et_rad = ((24.0 * 60.0) / np.pi) * SOLAR_CONSTANT * ird * (sha * np.sin(latitude) * np.sin(sol_dec) + np.cos(latitude) * np.cos(sol_dec) * np.sin(sha))
        
        # Calculate potential evaporation
        pe[ii] = 0.0023 * (tmean[ii] + 17.8) * np.sqrt(trange) * 0.408 * et_rad
        
    pe = np.maximum(0, pe)  # Ensure potential evaporation values are non-negative
    
    return pe
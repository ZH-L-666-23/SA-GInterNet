import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime

class CovariateDataProcessor:
    def __init__(self, raw_dir='data/raw/covariates', processed_dir='data/processed'):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
    
    def load_nc_file(self, filename):
        filepath = os.path.join(self.raw_dir, filename)
        ds = nc.Dataset(filepath, 'r')
        
        data_dict = {}
        for var_name in ds.variables:
            data_dict[var_name] = ds.variables[var_name][:]
        
        ds.close()
        return data_dict
    
    def load_csv_file(self, filename):
        filepath = os.path.join(self.raw_dir, filename)
        return pd.read_csv(filepath)
    
    def process_temperature(self, data_dict):
        temp = data_dict['temperature']
        lat = data_dict['latitude']
        lon = data_dict['longitude']
        time = data_dict['time']
        
        df_list = []
        for t_idx in range(len(time)):
            df = pd.DataFrame({
                'latitude': np.repeat(lat, len(lon)),
                'longitude': np.tile(lon, len(lat)),
                'temperature': temp[t_idx].flatten(),
                'timestamp': time[t_idx]
            })
            df_list.append(df)
        
        return pd.concat(df_list, ignore_index=True)
    
    def process_co2_flux(self, data_dict):
        flux = data_dict['co2_flux']
        lat = data_dict['latitude']
        lon = data_dict['longitude']
        time = data_dict['time']
        
        df_list = []
        for t_idx in range(len(time)):
            df = pd.DataFrame({
                'latitude': np.repeat(lat, len(lon)),
                'longitude': np.tile(lon, len(lat)),
                'co2_flux': flux[t_idx].flatten(),
                'timestamp': time[t_idx]
            })
            df_list.append(df)
        
        return pd.concat(df_list, ignore_index=True)
    
    def process_ndvi(self, data_dict):
        ndvi = data_dict['ndvi']
        lat = data_dict['latitude']
        lon = data_dict['longitude']
        time = data_dict['time']
        
        df_list = []
        for t_idx in range(len(time)):
            df = pd.DataFrame({
                'latitude': np.repeat(lat, len(lon)),
                'longitude': np.tile(lon, len(lat)),
                'ndvi': ndvi[t_idx].flatten(),
                'timestamp': time[t_idx]
            })
            df_list.append(df)
        
        return pd.concat(df_list, ignore_index=True)
    
    def process_pressure(self, data_dict):
        pressure = data_dict['pressure']
        lat = data_dict['latitude']
        lon = data_dict['longitude']
        time = data_dict['time']
        
        df_list = []
        for t_idx in range(len(time)):
            df = pd.DataFrame({
                'latitude': np.repeat(lat, len(lon)),
                'longitude': np.tile(lon, len(lat)),
                'pressure': pressure[t_idx].flatten(),
                'timestamp': time[t_idx]
            })
            df_list.append(df)
        
        return pd.concat(df_list, ignore_index=True)
    
    def resample_to_monthly(self, df):
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        monthly_df = df.groupby(['year', 'month', 'latitude', 'longitude']).mean().reset_index()
        return monthly_df.drop(columns=['timestamp', 'date'])
    
    def normalize_features(self, df, feature_columns):
        for col in feature_columns:
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
        return df
    
    def process_all_covariates(self):
        covariates = {}
        
        for filename in os.listdir(self.raw_dir):
            filepath = os.path.join(self.raw_dir, filename)
            
            if filename.endswith('.nc'):
                data_dict = self.load_nc_file(filename)
                
                if 'temperature' in data_dict:
                    temp_df = self.process_temperature(data_dict)
                    covariates['temperature'] = temp_df
                
                elif 'co2_flux' in data_dict:
                    flux_df = self.process_co2_flux(data_dict)
                    covariates['co2_flux'] = flux_df
                
                elif 'ndvi' in data_dict:
                    ndvi_df = self.process_ndvi(data_dict)
                    covariates['ndvi'] = ndvi_df
                
                elif 'pressure' in data_dict:
                    pressure_df = self.process_pressure(data_dict)
                    covariates['pressure'] = pressure_df
            
            elif filename.endswith('.csv'):
                df = self.load_csv_file(filename)
                var_name = filename.replace('.csv', '')
                covariates[var_name] = df
        
        if covariates:
            merged_df = covariates[list(covariates.keys())[0]]
            for key in list(covariates.keys())[1:]:
                merged_df = merged_df.merge(covariates[key], on=['year', 'month', 'latitude', 'longitude'], how='outer')
            
            merged_df = self.resample_to_monthly(merged_df)
            feature_cols = [col for col in merged_df.columns if col not in ['year', 'month', 'latitude', 'longitude']]
            merged_df = self.normalize_features(merged_df, feature_cols)
            
            output_path = os.path.join(self.processed_dir, 'covariates_monthly.csv')
            merged_df.to_csv(output_path, index=False)
            print(f"Covariate data processed and saved to {output_path}")
            return merged_df
        else:
            print("No covariate data files found")
            return None

if __name__ == '__main__':
    processor = CovariateDataProcessor()
    processor.process_all_covariates()
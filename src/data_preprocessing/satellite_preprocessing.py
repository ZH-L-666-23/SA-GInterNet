import os
import numpy as np
import pandas as pd
import h5py
from datetime import datetime
from scipy.interpolate import griddata

class SatelliteDataProcessor:
    def __init__(self, raw_dir='data/raw/satellite', processed_dir='data/processed'):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)
    
    def load_oco2_data(self, filename):
        filepath = os.path.join(self.raw_dir, filename)
        with h5py.File(filepath, 'r') as f:
            lat = f['/RetrievalGeometry/latitude'][:]
            lon = f['/RetrievalGeometry/longitude'][:]
            xco2 = f['/RetrievalResults/xco2'][:]
            xco2_uncertainty = f['/RetrievalResults/xco2_uncertainty'][:]
            time = f['/RetrievalGeometry/time'][:]
        
        df = pd.DataFrame({
            'latitude': lat,
            'longitude': lon,
            'xco2': xco2,
            'xco2_uncertainty': xco2_uncertainty,
            'timestamp': time
        })
        return df
    
    def load_oco3_data(self, filename):
        filepath = os.path.join(self.raw_dir, filename)
        with h5py.File(filepath, 'r') as f:
            lat = f['/geolocation/latitude'][:]
            lon = f['/geolocation/longitude'][:]
            xco2 = f['/retrieval_results/xco2'][:]
            xco2_quality_flag = f['/retrieval_results/xco2_quality_flag'][:]
            time = f['/geolocation/time'][:]
        
        df = pd.DataFrame({
            'latitude': lat,
            'longitude': lon,
            'xco2': xco2,
            'quality_flag': xco2_quality_flag,
            'timestamp': time
        })
        return df
    
    def filter_by_quality(self, df, quality_threshold=0.7):
        if 'quality_flag' in df.columns:
            df = df[df['quality_flag'] >= quality_threshold]
        if 'xco2_uncertainty' in df.columns:
            df = df[df['xco2_uncertainty'] < 3.0]
        return df
    
    def remove_outliers(self, df, column='xco2', z_threshold=3):
        mean = df[column].mean()
        std = df[column].std()
        lower = mean - z_threshold * std
        upper = mean + z_threshold * std
        return df[(df[column] >= lower) & (df[column] <= upper)]
    
    def resample_to_monthly(self, df):
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        monthly_df = df.groupby(['year', 'month', 'latitude', 'longitude']).agg({
            'xco2': ['mean', 'std', 'count'],
            'xco2_uncertainty': 'mean'
        }).reset_index()
        monthly_df.columns = ['year', 'month', 'latitude', 'longitude', 'xco2_mean', 'xco2_std', 'xco2_count', 'xco2_uncertainty_mean']
        return monthly_df
    
    def interpolate_spatial(self, df, grid_resolution=1.0):
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        
        lat_grid = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
        lon_grid = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        points = df[['longitude', 'latitude']].values
        values = df['xco2_mean'].values
        
        interpolated = griddata(points, values, (lon_mesh, lat_mesh), method='cubic')
        
        grid_df = pd.DataFrame({
            'latitude': lat_mesh.flatten(),
            'longitude': lon_mesh.flatten(),
            'xco2_interpolated': interpolated.flatten()
        }).dropna()
        
        return grid_df
    
    def process_all_satellite_data(self):
        processed_data = []
        
        for filename in os.listdir(self.raw_dir):
            if filename.endswith('.h5'):
                if 'oco2' in filename.lower():
                    df = self.load_oco2_data(filename)
                elif 'oco3' in filename.lower():
                    df = self.load_oco3_data(filename)
                else:
                    continue
                
                df = self.filter_by_quality(df)
                df = self.remove_outliers(df)
                monthly_df = self.resample_to_monthly(df)
                processed_data.append(monthly_df)
        
        if processed_data:
            combined_df = pd.concat(processed_data, ignore_index=True)
            output_path = os.path.join(self.processed_dir, 'satellite_xco2_monthly.csv')
            combined_df.to_csv(output_path, index=False)
            print(f"Satellite data processed and saved to {output_path}")
            return combined_df
        else:
            print("No satellite data files found")
            return None

if __name__ == '__main__':
    processor = SatelliteDataProcessor()
    processor.process_all_satellite_data()
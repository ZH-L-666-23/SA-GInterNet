import os
import numpy as np
import pandas as pd

np.random.seed(42)

PROVINCES_BOUNDS = {
    'Hunan': {'lon_min': 108.8, 'lon_max': 114.2, 'lat_min': 24.6, 'lat_max': 30.1},
    'Hubei': {'lon_min': 108.3, 'lon_max': 116.1, 'lat_min': 29.0, 'lat_max': 33.3},
    'Anhui': {'lon_min': 114.8, 'lon_max': 119.6, 'lat_min': 29.3, 'lat_max': 34.6},
    'Jiangxi': {'lon_min': 113.8, 'lon_max': 118.4, 'lat_min': 24.5, 'lat_max': 30.1}
}

PROVINCES = ['Hunan', 'Hubei', 'Anhui', 'Jiangxi']
SAMPLES_PER_PROVINCE = 250

XCO2_RANGES = {
    'winter': (420, 428),
    'spring': (412, 420),
    'summer': (400, 410),
    'autumn': (410, 418)
}

YEAR_OFFSETS = {
    2017: 0.0,
    2018: 0.5,
    2019: 1.0,
    2020: 1.5
}

def get_season(month):
    if month in [1, 2, 12]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

def generate_covariates(lat, lon, month, year):
    month_factor = np.sin(2 * np.pi * month / 12)

    NTL = np.random.uniform(20, 80) + 10 * month_factor
    NTL = np.clip(NTL, 0, 100)

    TEM = 15 + 20 * np.sin(2 * np.pi * (month - 4) / 12) + np.random.uniform(-3, 3)
    if month in [12, 1, 2]:
        TEM = np.random.uniform(-5, 10)
    elif month in [6, 7, 8]:
        TEM = np.random.uniform(25, 35)

    SP = 1010 + np.random.uniform(-15, 15)

    PRE = 80 + 100 * max(0, month_factor) + np.random.uniform(-20, 20)
    if month in [6, 7, 8]:
        PRE = 150 + np.random.uniform(-30, 30)
    PRE = np.clip(PRE, 20, 300)

    ET = 40 + 30 * max(0, month_factor) + np.random.uniform(-10, 10)
    if month in [6, 7, 8]:
        ET = 80 + np.random.uniform(-15, 15)
    elif month in [12, 1, 2]:
        ET = 20 + np.random.uniform(-5, 5)
    ET = np.clip(ET, 10, 120)

    UW = np.random.uniform(-8, 8) + 2 * month_factor
    VW = np.random.uniform(-6, 6)

    DSR = 180 + 100 * max(0, month_factor) + np.random.uniform(-30, 30)
    if month in [6, 7, 8]:
        DSR = 280 + np.random.uniform(-40, 40)
    elif month in [12, 1, 2]:
        DSR = 80 + np.random.uniform(-20, 20)
    DSR = np.clip(DSR, 50, 400)

    AOD = 0.2 + 0.15 * max(0, -month_factor) + np.random.uniform(-0.1, 0.1)
    if month in [6, 7, 8]:
        AOD = 0.4 + np.random.uniform(-0.15, 0.15)
    AOD = np.clip(AOD, 0.05, 0.8)

    NDVI_val = 0.4 + 0.3 * max(0, month_factor) + np.random.uniform(-0.1, 0.1)
    if month in [6, 7, 8]:
        NDVI_val = 0.7 + np.random.uniform(-0.1, 0.1)
    elif month in [12, 1, 2]:
        NDVI_val = 0.2 + np.random.uniform(-0.05, 0.05)
    NDVI_val = np.clip(NDVI_val, 0.1, 0.9)

    CO = 0.8 + 0.2 * np.random.uniform(-1, 1)
    if month in [12, 1, 2]:
        CO = 1.2 + np.random.uniform(-0.3, 0.3)
    elif month in [6, 7, 8]:
        CO = 0.6 + np.random.uniform(-0.2, 0.2)
    CO = np.clip(CO, 0.3, 2.0)

    NO2 = 15 + 8 * max(0, -month_factor) + np.random.uniform(-3, 3)
    if month in [12, 1, 2]:
        NO2 = 25 + np.random.uniform(-5, 5)
    elif month in [6, 7, 8]:
        NO2 = 10 + np.random.uniform(-3, 3)
    NO2 = np.clip(NO2, 5, 40)

    SO2 = 8 + 5 * max(0, -month_factor) + np.random.uniform(-2, 2)
    if month in [12, 1, 2]:
        SO2 = 15 + np.random.uniform(-4, 4)
    elif month in [6, 7, 8]:
        SO2 = 5 + np.random.uniform(-2, 2)
    SO2 = np.clip(SO2, 2, 25)

    O3 = 45 + 25 * max(0, month_factor) + np.random.uniform(-8, 8)
    if month in [6, 7, 8]:
        O3 = 80 + np.random.uniform(-15, 15)
    elif month in [12, 1, 2]:
        O3 = 35 + np.random.uniform(-8, 8)
    O3 = np.clip(O3, 20, 120)

    DEM = 200 + np.random.uniform(-150, 800)
    DEM = np.clip(DEM, 0, 2100)

    return {
        'NTL': round(NTL, 2),
        'TEM': round(TEM, 2),
        'SP': round(SP, 2),
        'PRE': round(PRE, 2),
        'ET': round(ET, 2),
        'UW': round(UW, 2),
        'VW': round(VW, 2),
        'DSR': round(DSR, 2),
        'AOD': round(AOD, 3),
        'NDVI': round(NDVI_val, 3),
        'CO': round(CO, 3),
        'NO2': round(NO2, 2),
        'SO2': round(SO2, 2),
        'O3': round(O3, 2),
        'DEM': round(DEM, 2)
    }

def generate_xco2(month, year):
    season = get_season(month)
    xco2_min, xco2_max = XCO2_RANGES[season]
    year_offset = YEAR_OFFSETS[year]

    xco2 = np.random.uniform(xco2_min + year_offset, xco2_max + year_offset)
    xco2 = np.clip(xco2, 400, 428)

    return round(xco2, 2)

def generate_sample_id(province, idx):
    province_code = {'Hunan': 'HN', 'Hubei': 'HB', 'Anhui': 'AH', 'Jiangxi': 'JX'}
    return f"{province_code[province]}_{idx:04d}"

def get_days_in_month(month, year):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 29
        else:
            return 28

def generate_province_data(province, year, month, start_idx):
    bounds = PROVINCES_BOUNDS[province]
    data = []
    days_in_month = get_days_in_month(month, year)

    for i in range(SAMPLES_PER_PROVINCE):
        sample_id = generate_sample_id(province, start_idx + i)
        lon = np.random.uniform(bounds['lon_min'], bounds['lon_max'])
        lat = np.random.uniform(bounds['lat_min'], bounds['lat_max'])
        day = np.random.randint(1, days_in_month + 1)
        date = f"{year}-{month:02d}-{day:02d}"

        xco2 = generate_xco2(month, year)
        covariates = generate_covariates(lat, lon, month, year)

        row = {
            'sample_id': sample_id,
            'lon': round(lon, 4),
            'lat': round(lat, 4),
            'date': date,
            'province': province,
            'XCO2': xco2
        }
        row.update(covariates)
        data.append(row)

    return data

def generate_monthly_file(year, month, output_dir):
    all_data = []
    global_idx = 0

    for province in PROVINCES:
        province_data = generate_province_data(province, year, month, global_idx)
        all_data.extend(province_data)
        global_idx += SAMPLES_PER_PROVINCE

    df = pd.DataFrame(all_data)

    columns = ['sample_id', 'lon', 'lat', 'date', 'province', 'XCO2',
               'NTL', 'TEM', 'SP', 'PRE', 'ET', 'UW', 'VW', 'DSR', 'AOD',
               'NDVI', 'CO', 'NO2', 'SO2', 'O3', 'DEM']
    df = df[columns]

    filename = f"CO2_Hunan_Hubei_Anhui_Jiangxi_{year}-{month:02d}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)

    return filepath, len(df)

def generate_all_files(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("Generating 48 CSV files for 2017-2020...")
    print("=" * 60)

    file_count = 0
    for year in range(2017, 2021):
        for month in range(1, 13):
            filepath, n_rows = generate_monthly_file(year, month, output_dir)
            file_count += 1
            print(f"[{file_count:02d}/48] {os.path.basename(filepath)} - {n_rows} samples")

    print("=" * 60)
    print(f"Successfully generated 48 CSV files in {output_dir}")

    verify_files(output_dir)

def verify_files(output_dir):
    print("\nVerifying generated files...")
    print("=" * 60)

    all_files = []
    for year in range(2017, 2021):
        for month in range(1, 13):
            filename = f"CO2_Hunan_Hubei_Anhui_Jiangxi_{year}-{month:02d}.csv"
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                all_files.append((filename, df))

                xco2_col = df['XCO2']
                season = get_season(month)
                xco2_min, xco2_max = XCO2_RANGES[season]

                print(f"{filename}:")
                print(f"  - Samples: {len(df)}")
                print(f"  - XCO2 range: {xco2_col.min():.2f} - {xco2_col.max():.2f}")
                print(f"  - Expected range ({season}): {xco2_min} - {xco2_max}")
                print(f"  - Provinces: {df['province'].value_counts().to_dict()}")
                print()

    expected_files = 48
    actual_files = len(all_files)
    print(f"Expected: {expected_files} files, Generated: {actual_files} files")
    print("Verification complete!")

if __name__ == '__main__':
    output_dir = r'c:\Users\LoveYourself\Desktop\STCGSANproject\data\processed'
    generate_all_files(output_dir)
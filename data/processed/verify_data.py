import os
import numpy as np
import pandas as pd

output_dir = r'c:\Users\LoveYourself\Desktop\STCGSANproject\data\processed'

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

print("Verifying XCO2 ranges by year and season...")
print("=" * 80)

for year in range(2017, 2021):
    year_offset = YEAR_OFFSETS[year]
    print(f"\nYear {year} (offset: +{year_offset} ppm):")
    for month in [1, 4, 7, 10]:
        filename = f"CO2_Hunan_Hubei_Anhui_Jiangxi_{year}-{month:02d}.csv"
        filepath = os.path.join(output_dir, filename)
        df = pd.read_csv(filepath)
        xco2_mean = df['XCO2'].mean()
        xco2_min = df['XCO2'].min()
        xco2_max = df['XCO2'].max()
        season = get_season(month)
        expected_min, expected_max = XCO2_RANGES[season]
        expected_min += year_offset
        expected_max += year_offset

        status = "OK" if xco2_min >= 400 and xco2_max <= 428 else "FAIL"
        print(f"  Month {month:02d} ({season:6s}): XCO2 [{xco2_min:.2f}, {xco2_max:.2f}], "
              f"mean={xco2_mean:.2f}, expected=[{expected_min:.1f}, {expected_max:.1f}] {status}")

print("\n" + "=" * 80)
print("Province distribution verification:")
for year in [2017]:
    for month in [1]:
        filename = f"CO2_Hunan_Hubei_Anhui_Jiangxi_{year}-{month:02d}.csv"
        filepath = os.path.join(output_dir, filename)
        df = pd.read_csv(filepath)
        print(f"\n{filename}:")
        print(df['province'].value_counts())
        print(f"Total samples: {len(df)}")

print("\n" + "=" * 80)
print("Column structure verification:")
filepath = os.path.join(output_dir, f"CO2_Hunan_Hubei_Anhui_Jiangxi_2017-01.csv")
df = pd.read_csv(filepath)
expected_cols = ['sample_id', 'lon', 'lat', 'date', 'province', 'XCO2',
                'NTL', 'TEM', 'SP', 'PRE', 'ET', 'UW', 'VW', 'DSR', 'AOD',
                'NDVI', 'CO', 'NO2', 'SO2', 'O3', 'DEM']
print(f"Columns match: {list(df.columns) == expected_cols}")
print(f"Columns: {list(df.columns)}")
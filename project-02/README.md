# 🚗 Traffic Volume Analysis

## 📌 Project Overview

This project analyzes hourly westbound traffic volume on Interstate 94 (I-94) between Minneapolis and St. Paul, Minnesota. The dataset contains traffic, weather, holiday, and time-related information collected from 2012 to 2018.

The main goal of this project is to explore the factors associated with variations in hourly traffic volume and identify meaningful patterns in traffic behavior across different times, weather conditions, and holidays.

### 📊 Dataset

The dataset contains **48,204 hourly observations** and **8 input features**, with `traffic_volume` as the target variable.

The data were collected from Minnesota Department of Transportation Automatic Traffic Recorder (ATR) station 301, located approximately midway between Minneapolis and St. Paul.

### 🧩 Features

| Feature | Type | Description |
|---|---|---|
| `holiday` | Categorical | U.S. national holidays and the Minnesota State Fair |
| `temp` | Continuous | Average temperature in Kelvin |
| `rain_1h` | Continuous | Amount of rainfall during the hour (mm) |
| `snow_1h` | Continuous | Amount of snowfall during the hour (mm) |
| `clouds_all` | Integer | Percentage of cloud cover |
| `weather_main` | Categorical | Short description of the prevailing weather condition |
| `weather_description` | Categorical | More detailed description of the weather condition |
| `date_time` | Date/Time | Local date and time of the observation |
| `traffic_volume` | Integer | Hourly westbound traffic volume reported by ATR station 301 |


🧹 Data Cleaning & Feature Engineering

The raw data was cleaned and enriched in a separate, fully reproducible notebook that starts from the raw CSV and ends with assertion checks.

Main fixes

Merged 7,612 duplicate-timestamp rows into one row per hour (48,204 → 40,575 rows).
Fixed impossible values: temperature of 0 K, two "stuck" temperature values (276.793 K, 297.888 K) and a 9,831 mm/h rainfall outlier.
Normalized weather label casing and rebuilt the incomplete holiday column for all 24 hours.
Result: 0 missing cells; the target traffic_volume is untouched.

New features (30): calendar and cyclical time features, weekday-only rush hour, holiday / day-before / day-after flags, weather flags and log-transformed precipitation.

📄 Full details, evidence and decisions: docs/DATA_CLEANING_REPORT.md 📓 Notebook: traffic_cleaning_feature_engineering_final.ipynb 📦 Clean data: Metro_Traffic_Clean_FE.csv (columns listed in feature_lists.json; do not train on qa_* columns)ll include duplicated timestamps and the anomalies above, so they should be re-run on the cleaned file.

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


## 🧹 Data Cleaning & Feature Engineering

The raw dataset was cleaned and prepared through a separate reproducible data-cleaning and feature-engineering process.

### 🔧 Data Cleaning

- Merged duplicate hourly records, reducing the dataset from **48,204 to 40,575 observations**.
- Identified and corrected invalid temperature values and converted temperature from **Kelvin to Celsius**.
- Detected and corrected an extreme rainfall outlier (**9,831.3 mm/h**).
- Normalized inconsistent weather labels.
- Removed the unreliable `holiday` feature due to excessive missing values.
- Preserved the original `traffic_volume` target without modification.
- Documented missing time periods and suspicious low traffic values without artificially changing the data.

### ⚙️ Feature Engineering

New features were created to capture important traffic patterns, including:

- Calendar features such as **year, month, weekday, hour, weekend, and season**.
- **Cyclical time features** for hour, weekday, and month.
- Weekday **rush-hour indicators**.
- Weather-related features such as **temperature, freezing, rain, snow, and precipitation**.
- Log-transformed precipitation features.

The final dataset contains **40,575 observations, 23 model features, the `traffic_volume` target, and 4 QA columns**. QA columns are used only for validation and are not used for model training.

📓 **Notebook:** `data_cleaning.csv`

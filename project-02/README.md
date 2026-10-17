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

📓 **Cleaned Dataset:** `data/data_cleaning.csv`

## 📈 Linear Regression Modeling

The cleaned hourly data were modeled with ordinary least-squares linear regression on the 23 engineered features (numeric inputs standardized, `season` one-hot encoded, the `qa_` columns never used). To respect the time structure of the data, the oldest 80% of the hours (2012-10-02 to 2017-10-26, 32,460 rows) were used for training and the newest 20% (to 2018-09-30, 8,115 rows) for testing, with the scaler and encoder fitted on the training part only. The baseline model reaches a test R² of 0.857 (RMSE ≈ 745 vehicles/hour, against a standard deviation of ≈ 1,985 for the target), with almost identical training and test scores. Residual diagnostics, however, show a clear weakness: the errors follow a daily wave, consecutive hours have strongly correlated errors (Durbin-Watson 0.85), and the precipitation flags are almost perfectly collinear (VIF > 1,000). A single straight line cannot draw the double-peak daily traffic curve, so four optional variants were tested. Adding the second daily harmonic of the hour together with weekday/weekend interactions (V4) clearly gave the best model, cutting the test RMSE by about 19%.

| Model | Model inputs | Test R² | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| **A** – Baseline (23 features) | 25 | 0.857 | 744.9 | 559.2 |
| **V1** – Cyclical encoding only (raw `hour`, `month`, `day_of_week` dropped) | 22 | 0.838 | 792.5 | 605.5 |
| **V2** – Log-transformed target (`log1p`) | 25 | 0.620 | 1213.7 | 832.1 |
| **V3** – Trained without the 46 flagged low-volume hours | 25 | 0.857 | 744.6 | 558.9 |
| **V4** – Hour harmonics × weekend interactions | 31 | **0.906** | **603.9** | **426.0** |

*RMSE and MAE are in vehicles per hour, computed on the chronological test set.*

The decision to leave weather labels out of the features was also tested. The 37 textual weather descriptions were merged into six groups with a similar effect on driving conditions (clear/cloudy, reduced visibility such as mist, fog, haze and smoke, light rain/drizzle, heavy rain, snow/ice, and severe storms covering thunderstorms and squalls). Adding these groups to the baseline changed the test RMSE only from 744.9 to 744.0, and a partial F-test found no significant gain (p = 0.30); removing all weather information costs only about 2 RMSE units, so hour of day and weekday dominate hourly traffic in a linear model. These results support the choice to exclude the labels, with two caveats: a linear model cannot capture interactions such as fog mattering more during rush hour, and the severe-storm group has only 239 hours. 📓 **Notebook:** `notebooks/traffic_volume_prediction.ipynb` (sections 5,6).

## 🌲 Random Forest Modeling

The final notebook section (Section 7) implements **Random Forest regression** using the same chronological training/test split and the same 23 engineered input features as the linear models. The four `qa_` columns are excluded. The 32,460 earlier observations are used for training and the latest 8,115 observations are reserved for testing, so model selection does not use the final test period.

### Pipeline and baseline

The Random Forest pipeline passes numeric columns through without scaling and one-hot encodes the categorical `season` feature. Preprocessing is fitted inside the training pipeline. The baseline uses 200 trees, unlimited tree depth, a minimum of one sample per leaf, and all features available at each split. A fixed random seed (`random_state=42`) makes the run reproducible, while `n_jobs=-1` enables parallel processing.

### Parameter comparison

Five parameter settings were compared on five chronological, calendar-based validation windows contained within the training data. Each validation block is predicted using only observations that come before that block. The windows run from April 2015 through October 2017; the final window ends at the training-period boundary. The table reports the mean across the five windows.

| Configuration | `n_estimators` | `max_depth` | `min_samples_leaf` | `max_features` | Mean R² | Mean MAE | Mean RMSE | RMSE std. dev. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Larger leaves (selected)** | 200 | None | 5 | 1.0 | **0.935** | **300.286** | **499.895** | 54.700 |
| Fewer features | 200 | None | 1 | 0.7 | 0.931 | 313.033 | 513.928 | 55.927 |
| More trees | 400 | None | 1 | 1.0 | 0.929 | 318.566 | 522.809 | 59.532 |
| Baseline | 200 | None | 1 | 1.0 | 0.929 | 319.167 | 523.717 | 59.289 |
| Limited depth | 200 | 25 | 1 | 1.0 | 0.929 | 319.346 | 524.111 | 60.659 |

The selected configuration is the one with the lowest mean validation RMSE: 200 trees, unlimited depth, `min_samples_leaf=5`, and `max_features=1.0`. Increasing the minimum leaf size regularizes the model and improves its validation performance among the tested configurations. This is a small, targeted parameter comparison rather than an exhaustive search.

### Test-set comparison

After selecting the parameters, the model is refit on the full training period and evaluated on the held-out test period. Scores are in vehicles per hour, except R².

| Model | Train R² | Test R² | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|
| Linear Regression A | 0.858 | 0.857 | 744.872 | 559.248 |
| Linear Regression V4 | 0.907 | 0.906 | 603.882 | 425.975 |
| Random Forest baseline | 0.993 | 0.937 | 493.966 | 286.556 |
| **Random Forest tuned** | **0.969** | **0.942** | **476.296** | **270.062** |

The tuned Random Forest achieves the strongest test results among these four models. Its test RMSE is about 21% lower than Linear Regression V4, and its MAE is about 36% lower. Compared with the baseline forest, tuning reduces test RMSE by about 18 vehicles/hour and test MAE by about 16 vehicles/hour. The lower training score and improved test score after increasing `min_samples_leaf` are consistent with reduced overfitting.

### Feature importance and error analysis

Permutation importance measures the increase in test RMSE after shuffling an input column; the individual-feature calculation uses 10 repeats. The largest individual contributions are `hour_cos` (1,513.810 RMSE units), `hour` (506.806), `hour_sin` (212.592), and `is_rush_hour` (157.176). Because these time features overlap in the information they encode, the notebook also shuffles related features as groups. The grouped increases in RMSE are 2,165.29 for hour/commute features and 637.38 for weekday/weekend features. Weather groups have much smaller measured effects in this test evaluation: 30.62 for temperature/freezing, 0.45 for precipitation, and 2.52 for cloud cover. Importance indicates predictive reliance in this model and dataset; it does not establish causation.

The notebook also examines errors by hour, weekday/weekend, and season. On the test set, summer has the lowest seasonal MAE (202.9 vehicles/hour), while winter has the highest (332.0). Weekend bias is −50.1 vehicles/hour and spring bias is −71.0, where bias is defined as actual minus predicted; negative values indicate average overprediction. These breakdowns show that overall scores do not describe performance equally well for every time segment.

📓 **Notebook:** `notebooks/traffic_volume_prediction.ipynb` (Section 7).

## ⚡ XGBoost & LightGBM Modeling

XGBoost and LightGBM were implemented as gradient boosting baselines and evaluated using the same chronological train-test split and 23 engineered input features as the previous models. The preprocessing pipeline passes numeric features through and one-hot encodes `season`, resulting in **25 model inputs after preprocessing**. Both models were evaluated using R², RMSE, and MAE.

### Model setup

Both models used 300 boosting estimators, a learning rate of 0.1, and a fixed random seed (`random_state=42`) for reproducibility. XGBoost used a maximum tree depth of 6, while LightGBM used 31 leaves with no explicit maximum depth. These are baseline configurations; extensive hyperparameter tuning and final model selection are outside the scope of this section.

### Test-set results

| Model | Test R² | Test RMSE | Test MAE |
|---|---:|---:|---:|
| XGBoost baseline | 0.9420 | 474.352 | 274.293 |
| LightGBM baseline | **0.9429** | **470.897** | **269.214** |

LightGBM slightly outperformed XGBoost, achieving a lower RMSE and MAE and a marginally higher R². The performance difference is relatively small, so these results alone do not establish which model should be selected as the final model.

### Feature importance

Permutation importance was calculated on the test set using 10 repeats and the increase in RMSE after shuffling each feature. The results show that **time-related features dominate predictions in both models**.

For XGBoost, the most influential features were `hour_cos` (1,048.44), `hour` (860.91), and `day_of_week` (393.57). For LightGBM, the top three were `hour` (931.91), `hour_cos` (682.82), and `day_of_week` (291.86). The `is_rush_hour` feature had a larger importance score in LightGBM (266.06) than in XGBoost (115.98), while temperature had a moderate contribution in both models.

These findings reinforce the importance of daily and weekly traffic patterns. Permutation importance reflects how much a model relies on a feature for prediction; it does not establish a causal relationship.

📓 **Notebook:** `notebooks/traffic_volume_prediction.ipynb` (section 8).

## 🔬 Hyperparameter Tuning + Model Evaluation

Building upon the tree-based and gradient boosting baselines from Sections 7 and 8, an end-to-end Bayesian optimization and diagnostic framework was implemented in Section 9 (`notebooks/traffic_volume_prediction.ipynb`). This final phase focused on automated hyperparameter tuning using Optuna, multi-metric model benchmarking, and granular residual diagnostics to select the best production model.

### Optuna Optimization Setup

To maximize predictive accuracy without data leakage, automated hyperparameter tuning was conducted for **XGBoost**, **LightGBM**, and **Random Forest** on the chronological training set (**32,460 observations**) and evaluated on the held-out test period (**8,115 observations**).

- **Optimization Engine:** Optuna configured with a Tree-structured Parzen Estimator (TPE) sampler (`seed=42`) over **300 trials per model**.
- **Validation Strategy:** **5-fold Time-Series Cross-Validation** (`TimeSeriesSplit`), preserving temporal order across every training and validation fold.
- **Leak-Free Preprocessing:** Each estimator was encapsulated inside a full preprocessing pipeline (passing numerical features through and one-hot encoding the categorical `season` feature), resulting in **25 input features after preprocessing**.
- **Objective Metric:** Minimizing mean Cross-Validation Root Mean Squared Error (**CV RMSE**).
- **Dependency Note:** To render Optuna's live progress bar (`show_progress_bar=True`), ensure that the optional graphical dependency `tqdm` is installed in your environment (`pip install tqdm`).

### Optimal Hyperparameter Configurations

Across 300 trials per algorithm, Optuna identified the following optimal configurations:

- **Random Forest (Best Overall):** `n_estimators=800`, `max_depth=16`, `min_samples_split=9`, `min_samples_leaf=4`, `max_features='sqrt'`, `bootstrap=True`, `max_samples=0.9070`, `ccp_alpha=0.0`.
- **XGBoost:** `n_estimators=300`, `max_depth=7`, `learning_rate=0.0145`, `min_child_weight=20`, `subsample=0.6939`, `colsample_bytree=0.8202`, `colsample_bylevel=0.6365`, `gamma=0.0548`, `reg_alpha=0.0`, `reg_lambda=0.1148`, `max_delta_step=0`.
- **LightGBM:** `n_estimators=400`, `max_depth=9`, `num_leaves=53`, `learning_rate=0.0138`, `min_child_samples=71`, `subsample=0.6054`, `subsample_freq=4`, `colsample_bytree=0.6002`, `reg_alpha=0.0074`, `reg_lambda=0.1610`, `min_split_gain=0.1251`.

### Optimized Model Comparison

After tuning, each model was refit on the complete training set and benchmarked on the chronological test set across multiple regression metrics:

| Model | Features | Train R² | Test R² | CV RMSE | Train RMSE | Test RMSE | Test MAE | Test MAPE (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost (Tuned) | 25 | 0.9535 | 0.9441 | 485.54 | 428.94 | 465.84 | 265.25 | 13.42 |
| LightGBM (Tuned) | 25 | 0.9543 | 0.9439 | 484.62 | 425.29 | 466.56 | 263.71 | 13.37 |
| **Random Forest (Tuned)** | **25** | **0.9617** | **0.9446** | **484.27** | **388.94** | **463.76** | **263.44** | **12.59** |

*RMSE and MAE are reported in vehicles per hour.*

**Key Takeaways:**
- **Overall Winner:** **Random Forest emerged as the best-performing model** across every single evaluation metric, achieving the lowest CV RMSE (**484.27**), lowest Test RMSE (**463.76**), lowest Test MAE (**263.44**), lowest Test MAPE (**12.59%**), and highest Test R² (**0.9446**).
- **Gains Over Baselines:** Compared to the earlier manual baselines in Sections 7 and 8, Optuna tuning reduced Test RMSE across all models—improving Random Forest from **493.97** (baseline) and **476.30** (manual grid) down to **463.76**, XGBoost from **474.35** to **465.84**, and LightGBM from **470.90** to **466.56**.
- **Model Convergence:** While LightGBM edged out XGBoost in CV RMSE (**484.62** vs. **485.54**) and MAE (**263.71** vs. **265.25**), XGBoost achieved a slightly lower Test RMSE (**465.84** vs. **466.56**). Overall, Bayesian optimization substantially narrowed the performance gap between bagging and boosting architectures.

### Optimization Analysis & Residual Diagnostics

To interpret the tuning behavior and verify out-of-sample reliability for the winning **Random Forest** model, two sets of visual diagnostics were generated:

#### 1. Optuna Optimization Visuals
- **Optimization History:** The search started at an initial CV RMSE near **494** and achieved rapid convergence within the first **30 to 50 trials**, stabilizing into a consistent plateau at **484.27**.
- **Hyperparameter Importances:** Variance-based importance analysis revealed that **`max_features`** (**0.80 / 80%**) and **`bootstrap`** (**0.15 / 15%**) drove **95%** of total tuning gains. Restricting split candidates (`max_features='sqrt'`) and enabling bootstrap resampling (`bootstrap=True`) effectively decorrelated individual trees. Structural parameters played a minor secondary role: `min_samples_split` (**0.02**), `ccp_alpha` (**0.01**), `max_depth` (**0.01**), `min_samples_leaf` (**<0.01**), and `n_estimators` (**<0.01**).
- **Parallel Coordinates:** Multi-dimensional trajectory lines confirmed that top-performing trials consistently clustered around `bootstrap=True`, `max_features='sqrt'`, `max_depth` between **16 and 20**, and minimal post-pruning (`ccp_alpha ≈ 0.0`).

#### 2. Six-Panel Residual Diagnostics Dashboard
- **Actual vs. Predicted:** Test predictions align tightly along the 45-degree perfect-prediction line across the entire traffic spectrum (**0 to 7,000 vehicles/hour**), showing no saturation at peak volumes.
- **Residuals vs. Predicted:** Residuals scatter symmetrically around the zero baseline. Moderate variance widening appears at mid-to-high volumes (2,500–6,000 vehicles/hour), alongside a few negative outliers caused by sudden, unrecorded traffic drops.
- **Residual Distribution & Normal Q-Q Plot:** The error histogram forms a sharp, symmetric bell curve centered strictly at zero. The Q-Q plot closely tracks the theoretical normal distribution within ±2 standard deviations, with left-tail deviations reflecting isolated low-volume anomalies.
- **Residuals by Hour of Day:** Mean hourly residuals remain flat near zero across all 24 hours, proving that cyclical time transformations successfully eliminated diurnal bias. The ±1 standard deviation band widens naturally during active daytime commuting hours (06:00–18:00) and narrows overnight.
- **Last 7 Days of the Test Period:** A chronological overlay of actual vs. predicted volume over the final test week (**2018-09-24 to 2018-10-01**) demonstrates high precision in capturing both sharp bimodal weekday rush-hour peaks and softer unimodal weekend profiles.

📓 **Notebook:** `notebooks/traffic_volume_prediction.ipynb` (Section 9).
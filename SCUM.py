import os
from time import time
from typing import List, Tuple

# Set environment variables for StatsForecast optimization
os.environ["NIXTLA_NUMBA_RELEASE_GIL"] = "1"
os.environ["NIXTLA_NUMBA_CACHE"] = "1"
os.environ["NIXTLA_ID_AS_COL"] = "true"

import numpy as np
import polars as pl
from scipy.stats import norm
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoCES,
    DynamicOptimizedTheta,
    SeasonalNaive,
)

# User parameters and paths
data_path = '../data/your_excel_file.xlsx'  # Replace with your actual file path
output_dir = '../output/'  # Directory to save forecast Excel files
horizon = 12  # Forecast horizon (e.g., 12 months)
freq = 'M'    # Frequency ('M' for monthly data)
seasonality = 12  # Seasonality (e.g., 12 for monthly data)
level = [68.27]  # Confidence level for prediction intervals
quantiles = [0.025, 0.5, 0.975]  # Quantiles for forecasts

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Reading and preparing data
train_df = pl.read_excel(data_path)

# Concatenate 'org' and 'channel' to create 'unique_id'
train_df = train_df.with_columns(
    (pl.col('org') + '_' + pl.col('channel')).alias('unique_id')
)

# Ensure 'yymmdd' is of datetime type
train_df = train_df.with_columns(
    pl.col('yymmdd').str.strptime(pl.Date, fmt='%Y%m%d')  # Adjust '%Y%m%d' if needed
)

# Rename 'alerts' to 'y'
train_df = train_df.rename({'alerts': 'y'})

# Resample data to monthly frequency using alternative method
# Extract year and month from 'yymmdd'
train_df = train_df.with_columns(
    pl.col('yymmdd').dt.strftime('%Y-%m').alias('year_month')
)

# Parse 'year_month' back to a date (first day of the month)
train_df = train_df.with_columns(
    pl.col('year_month').str.strptime(pl.Date, fmt='%Y-%m').alias('ds')
)

# Group by 'unique_id' and 'ds', then aggregate 'y' by summing over the month
train_df = train_df.groupby(['unique_id', 'ds']).agg(
    pl.col('y').sum().alias('y')
)

# Select only the required columns
train_df = train_df.select(['unique_id', 'ds', 'y'])

# Now 'train_df' contains only 'unique_id', 'ds', and 'y'

def run_seasonal_naive(
    train_df: pl.DataFrame,
    horizon: int,
    freq: str,
    seasonality: int,
    level: List[int],
) -> Tuple[pl.DataFrame, float, str]:
    sf = StatsForecast(
        models=[SeasonalNaive(season_length=seasonality)],
        freq=freq,
        n_jobs=-1,
    )
    init_time = time()
    fcsts_df = sf.forecast(df=train_df, h=horizon, level=level)
    total_time = time() - init_time
    return fcsts_df, total_time, "SeasonalNaive"

def fcst_from_level_to_quantiles(
    fcst_df: pl.DataFrame, model_name: str, quantiles: List[float]
) -> pl.DataFrame:
    # Compute standard deviation from the 68.27% confidence interval
    z_68 = norm.ppf(0.5 + 68.27 / 200)
    fcst_df = fcst_df.with_columns(
        ((pl.col(f"{model_name}-hi-68.27") - pl.col(model_name)) / z_68).alias(f"std_{model_name}")
    )
    # Compute the quantiles
    z = norm.ppf(quantiles)
    for q, zq in zip(quantiles, z):
        q_col = f"{model_name}-q-{q}"
        fcst_df = fcst_df.with_columns(
            (pl.col(model_name) + zq * pl.col(f"std_{model_name}")).alias(q_col)
        )
    # Select the relevant columns
    q_cols = [f"{model_name}-q-{q}" for q in quantiles]
    fcst_df = fcst_df.select(["unique_id", "ds", model_name] + q_cols)
    return fcst_df

def ensemble_forecasts(
    fcsts_df: pl.DataFrame,
    quantiles: List[float],
    name_models: List[str],
    model_name: str,
) -> pl.DataFrame:
    # Compute the mean across the models
    fcsts_df = fcsts_df.with_columns(
        pl.mean([pl.col(name) for name in name_models]).alias(model_name)
    )
    # Compute the sigma for each model
    sigma_models = []
    for model in name_models:
        sigma_col = f"sigma_{model}"
        fcsts_df = fcsts_df.with_columns(
            (pl.col(f"{model}-hi-68.27") - pl.col(model)).alias(sigma_col)
        )
        sigma_models.append(sigma_col)
    # Compute the standard deviation of the ensemble
    fcsts_df = fcsts_df.with_columns(
        (
            sum([pl.col(sigma_col) ** 2 for sigma_col in sigma_models])
            / (len(sigma_models) ** 2)
        )
        .sqrt()
        .alias(f"std_{model_name}")
    )
    # Compute the quantiles
    z = norm.ppf(quantiles)
    for q, zq in zip(quantiles, z):
        q_col = f"{model_name}-q-{q}"
        fcsts_df = fcsts_df.with_columns(
            (pl.col(model_name) + zq * pl.col(f"std_{model_name}")).alias(q_col)
        )
    # Select the relevant columns
    q_cols = [f"{model_name}-q-{q}" for q in quantiles]
    fcsts_df = fcsts_df.select(["unique_id", "ds", model_name] + q_cols)
    return fcsts_df

def run_statistical_ensemble(
    train_df: pl.DataFrame,
    horizon: int,
    freq: str,
    seasonality: int,
    quantiles: List[float],
) -> Tuple[pl.DataFrame, float, str]:
    models = [
        AutoARIMA(season_length=seasonality),
        AutoETS(season_length=seasonality),
        AutoCES(season_length=seasonality),
        DynamicOptimizedTheta(season_length=seasonality),
    ]
    init_time = time()
    series_per_core = 15
    n_series = train_df.select(pl.col("unique_id")).n_unique()
    n_jobs = min(max(n_series // series_per_core, 1), os.cpu_count())
    sf = StatsForecast(
        models=models,
        freq=freq,
        n_jobs=n_jobs,
    )
    fcsts_df = sf.forecast(df=train_df, h=horizon, level=[68.27])
    name_models = [repr(model) for model in models]
    model_name = "StatisticalEnsemble"
    fcsts_df = ensemble_forecasts(
        fcsts_df,
        quantiles,
        name_models,
        model_name,
    )
    total_time = time() - init_time
    return fcsts_df, total_time, model_name

# Run Seasonal Naive model
fcst_df_sn, total_time_sn, model_name_sn = run_seasonal_naive(
    train​⬤

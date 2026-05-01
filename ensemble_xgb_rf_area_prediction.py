# ensemble_xgb_rf_area_prediction.py

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ============================================================
# 0. 경로 및 기본 설정
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_DATES = [
    "2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04",
    "2025-09-05", "2025-09-06", "2025-09-07", "2025-09-08",
    "2025-09-09", "2025-09-10", "2025-09-11", "2025-09-12",
    "2025-09-13"
]

TEST_DATE = "2025-09-14"

TIME_INDEX_COL = "time_index"
AREA_COL = "area"
VALUE_COL = "num_people"

FORECAST_HORIZON = 1

XGB_WEIGHT = 0.5
RF_WEIGHT = 0.5

RANDOM_STATE = 42

USE_CACHE = True
CACHE_PATH = os.path.join(OUTPUT_DIR, "cached_1min_data.csv")


# ============================================================
# 1. 10초 데이터를 1분 평균으로 변환
# ============================================================

def preprocess_to_1min(df, date_str):
    df = df.copy()

    df[TIME_INDEX_COL] = df[TIME_INDEX_COL].astype(np.int32)
    df[VALUE_COL] = df[VALUE_COL].astype(np.float32)

    df["minute_index"] = (df[TIME_INDEX_COL] - 1) // 6

    base_time = pd.to_datetime(date_str)
    df["timestamp"] = base_time + pd.to_timedelta(df["minute_index"], unit="m")

    df_1min = (
        df.groupby(["timestamp", AREA_COL], as_index=False)[VALUE_COL]
          .mean()
          .rename(columns={
              AREA_COL: "zone",
              VALUE_COL: "y"
          })
    )

    df_1min["y"] = df_1min["y"].astype(np.float32)

    return df_1min


def load_and_preprocess_files(raw_dir, dates):
    all_df = []
    usecols = [TIME_INDEX_COL, AREA_COL, VALUE_COL]

    for date in dates:
        compact_date = date.replace("-", "")
        file_path = os.path.join(
            raw_dir,
            f"area_count_time_full_{compact_date}.csv"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다:\n{file_path}")

        print("Loading:", file_path)

        df = pd.read_csv(file_path, usecols=usecols)
        df_1min = preprocess_to_1min(df, date)
        all_df.append(df_1min)

    return pd.concat(all_df, ignore_index=True)


def load_or_make_1min_data():
    if USE_CACHE and os.path.exists(CACHE_PATH):
        print("========== Load Cached 1-minute Data ==========")
        df = pd.read_csv(CACHE_PATH)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    print("========== Load & 1-minute Aggregation ==========")

    train_1min = load_and_preprocess_files(RAW_DIR, TRAIN_DATES)
    test_1min = load_and_preprocess_files(RAW_DIR, [TEST_DATE])

    df = pd.concat([train_1min, test_1min], ignore_index=True)

    if USE_CACHE:
        df.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")
        print("Cached saved:", CACHE_PATH)

    return df


# ============================================================
# 2. Feature Engineering
# ============================================================

def add_time_features(df):
    df = df.copy()

    df["minute"] = df["timestamp"].dt.minute.astype(np.int16)
    df["hour"] = df["timestamp"].dt.hour.astype(np.int16)
    df["dayofweek"] = df["timestamp"].dt.dayofweek.astype(np.int16)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype(np.float32)

    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60).astype(np.float32)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60).astype(np.float32)

    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype(np.float32)

    return df


def add_lag_rolling_features(df):
    df = df.copy()
    df = df.sort_values(["zone", "timestamp"])

    lag_list = [1, 2, 3, 5, 10, 15, 30, 60, 120, 180]
    rolling_list = [3, 5, 10, 15, 30, 60, 120, 180]

    for lag in lag_list:
        df[f"lag_{lag}"] = (
            df.groupby("zone")["y"]
              .shift(lag)
              .astype(np.float32)
        )

    shifted = df.groupby("zone")["y"].shift(1)

    for window in rolling_list:
        df[f"rolling_mean_{window}"] = (
            shifted.groupby(df["zone"])
                   .rolling(window)
                   .mean()
                   .reset_index(level=0, drop=True)
                   .astype(np.float32)
        )

        df[f"rolling_std_{window}"] = (
            shifted.groupby(df["zone"])
                   .rolling(window)
                   .std()
                   .reset_index(level=0, drop=True)
                   .astype(np.float32)
        )

    df["target"] = (
        df.groupby("zone")["y"]
          .shift(-FORECAST_HORIZON)
          .astype(np.float32)
    )

    df = df.dropna().reset_index(drop=True)

    return df


# ============================================================
# 3. 모델 정의
# ============================================================

def build_xgb_model():
    return XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )


def build_rf_model():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ============================================================
# 4. 평가 함수
# ============================================================

def evaluate_result(df, pred_col):
    y_true = df["actual"].values.astype(np.float32)
    y_pred = df[pred_col].values.astype(np.float32)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))
    ) * 100

    return {
        "MAE": mae,
        "MAPE(%)": mape,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 5. 메인 파이프라인
# ============================================================

def main():
    print("========== Path Check ==========")
    print("BASE_DIR:", BASE_DIR)
    print("RAW_DIR:", RAW_DIR)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    full_df = load_or_make_1min_data()

    print("1-minute data shape:", full_df.shape)

    print("\n========== Feature Engineering ==========")

    full_df = add_time_features(full_df)
    full_df = add_lag_rolling_features(full_df)

    full_df["date_str"] = full_df["timestamp"].dt.strftime("%Y-%m-%d")

    train_df = full_df[full_df["date_str"].isin(TRAIN_DATES)].copy()
    test_df = full_df[full_df["date_str"] == TEST_DATE].copy()

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    feature_cols = [
        col for col in train_df.columns
        if col not in [
            "timestamp",
            "date_str",
            "zone",
            "target"
        ]
    ]

    print("\nFeature columns:")
    print(feature_cols)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["target"].values.astype(np.float32)

    X_test = test_df[feature_cols].values.astype(np.float32)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # ========================================================
    # XGBoost
    # ========================================================

    print("\n========== Train XGBoost ==========")

    xgb_model = build_xgb_model()
    xgb_model.fit(X_train_scaled, y_train)

    xgb_pred = xgb_model.predict(X_test_scaled).astype(np.float32)

    # ========================================================
    # RandomForest
    # ========================================================

    print("\n========== Train RandomForest ==========")

    rf_model = build_rf_model()
    rf_model.fit(X_train_scaled, y_train)

    rf_pred = rf_model.predict(X_test_scaled).astype(np.float32)

    # ========================================================
    # 결과 정리
    # ========================================================

    result = test_df[["timestamp", "zone", "target"]].copy()
    result = result.rename(columns={"target": "actual"})

    result["xgb_pred"] = xgb_pred
    result["rf_pred"] = rf_pred

    result["xgb_pred"] = result["xgb_pred"].clip(lower=0)
    result["rf_pred"] = result["rf_pred"].clip(lower=0)

    result["ensemble_pred"] = (
        XGB_WEIGHT * result["xgb_pred"]
        + RF_WEIGHT * result["rf_pred"]
    ).clip(lower=0)

    # ========================================================
    # 평가
    # ========================================================

    print("\n========== Evaluation ==========")

    metrics = {
        "XGBoost": evaluate_result(result, "xgb_pred"),
        "RandomForest": evaluate_result(result, "rf_pred"),
        "Weighted Ensemble": evaluate_result(result, "ensemble_pred")
    }

    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df)

    zone_metrics = []

    for zone, g in result.groupby("zone"):
        m = evaluate_result(g, "ensemble_pred")
        m["zone"] = zone
        zone_metrics.append(m)

    zone_metrics_df = pd.DataFrame(zone_metrics)
    zone_metrics_df = zone_metrics_df[
        ["zone", "MAE", "MAPE(%)", "RMSE", "R2"]
    ]

    print("\n========== Zone Evaluation ==========")
    print(zone_metrics_df)

    # ========================================================
    # 저장
    # ========================================================

    result_path = os.path.join(
        OUTPUT_DIR,
        "ensemble_prediction_result_2025-09-14_xgb_rf.csv"
    )

    metrics_path = os.path.join(
        OUTPUT_DIR,
        "ensemble_metrics_2025-09-14_xgb_rf.csv"
    )

    zone_metrics_path = os.path.join(
        OUTPUT_DIR,
        "ensemble_zone_metrics_2025-09-14_xgb_rf.csv"
    )

    result.to_csv(result_path, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(metrics_path, encoding="utf-8-sig")
    zone_metrics_df.to_csv(zone_metrics_path, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(result_path)
    print(metrics_path)
    print(zone_metrics_path)


if __name__ == "__main__":
    main()

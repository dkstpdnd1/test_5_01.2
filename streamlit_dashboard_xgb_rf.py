# streamlit_dashboard_xgb_rf.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ============================================================
# 0. 기본 설정
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "ensemble_prediction_result_2025-09-14_xgb_rf.csv"
)

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "ensemble_metrics_2025-09-14_xgb_rf.csv"
)

ZONE_METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "ensemble_zone_metrics_2025-09-14_xgb_rf.csv"
)


# ============================================================
# 1. 데이터 로드
# ============================================================

@st.cache_data
def load_data():
    result = pd.read_csv(RESULT_FILE)
    metrics = pd.read_csv(METRICS_FILE, index_col=0)
    zone_metrics = pd.read_csv(ZONE_METRICS_FILE)

    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["date"] = result["timestamp"].dt.date
    result["time"] = result["timestamp"].dt.strftime("%H:%M")

    return result, metrics, zone_metrics


def calc_error_columns(df):
    df = df.copy()

    pred_cols = {
        "xgb": "xgb_pred",
        "rf": "rf_pred",
        "ensemble": "ensemble_pred"
    }

    for model_name, col in pred_cols.items():
        df[f"{model_name}_abs_error"] = np.abs(df["actual"] - df[col])
        df[f"{model_name}_ape"] = (
            np.abs(df["actual"] - df[col]) /
            np.maximum(np.abs(df["actual"]), 1)
        ) * 100

    return df


# ============================================================
# 2. Streamlit 화면 설정
# ============================================================

st.set_page_config(
    page_title="XGBoost + RandomForest 공항 구역별 예측 대시보드",
    layout="wide"
)

st.title("XGBoost + RandomForest 가중 앙상블 예측 대시보드")
st.caption("1분 평균 데이터 기반 / LSTM 제외 / 2025-09-14 실제값 비교")


# ============================================================
# 3. 파일 확인
# ============================================================

missing_files = []

for file in [RESULT_FILE, METRICS_FILE, ZONE_METRICS_FILE]:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    st.error("필요한 출력 파일을 찾을 수 없습니다.")
    for file in missing_files:
        st.code(file)

    st.info("먼저 아래 모델 코드를 실행해서 결과 CSV를 생성해야 합니다.")
    st.code("py -3.10 ensemble_xgb_rf_area_prediction.py", language="powershell")
    st.stop()


# ============================================================
# 4. 데이터 준비
# ============================================================

result, metrics, zone_metrics = load_data()
result = calc_error_columns(result)

zones = sorted(result["zone"].unique())

model_options = {
    "가중 앙상블": "ensemble_pred",
    "XGBoost": "xgb_pred",
    "RandomForest": "rf_pred"
}

error_model_map = {
    "가중 앙상블": "ensemble_ape",
    "XGBoost": "xgb_ape",
    "RandomForest": "rf_ape"
}


# ============================================================
# 5. 사이드바
# ============================================================

st.sidebar.header("필터 설정")

selected_zone = st.sidebar.selectbox(
    "구역 선택",
    zones
)

selected_models = st.sidebar.multiselect(
    "그래프에 표시할 모델",
    list(model_options.keys()),
    default=["가중 앙상블", "XGBoost", "RandomForest"]
)

time_min = result["timestamp"].min().time()
time_max = result["timestamp"].max().time()

selected_time_range = st.sidebar.slider(
    "시간 범위",
    min_value=time_min,
    max_value=time_max,
    value=(time_min, time_max)
)

show_raw_table = st.sidebar.checkbox(
    "예측 데이터 테이블 표시",
    value=False
)

show_heatmap = st.sidebar.checkbox(
    "전체 구역 히트맵 표시",
    value=True
)


# ============================================================
# 6. 필터 적용
# ============================================================

zone_df = result[result["zone"] == selected_zone].copy()

zone_df = zone_df[
    (zone_df["timestamp"].dt.time >= selected_time_range[0]) &
    (zone_df["timestamp"].dt.time <= selected_time_range[1])
].copy()


# ============================================================
# 7. 전체 모델 성능 비교
# ============================================================

st.subheader("전체 모델 성능 비교")

if "Weighted Ensemble" in metrics.index:
    ensemble_metrics = metrics.loc["Weighted Ensemble"]
else:
    ensemble_metrics = metrics.iloc[-1]

metric_cols = st.columns(4)

metric_cols[0].metric("앙상블 MAE", f"{ensemble_metrics['MAE']:.3f}")
metric_cols[1].metric("앙상블 MAPE", f"{ensemble_metrics['MAPE(%)']:.2f}%")
metric_cols[2].metric("앙상블 RMSE", f"{ensemble_metrics['RMSE']:.3f}")
metric_cols[3].metric("앙상블 R²", f"{ensemble_metrics['R2']:.4f}")

st.dataframe(
    metrics.style.format({
        "MAE": "{:.3f}",
        "MAPE(%)": "{:.2f}",
        "RMSE": "{:.3f}",
        "R2": "{:.4f}"
    }),
    use_container_width=True
)


# ============================================================
# 8. 구역별 성능
# ============================================================

st.subheader("구역별 앙상블 성능")

selected_zone_metric = zone_metrics[zone_metrics["zone"] == selected_zone]

if not selected_zone_metric.empty:
    z = selected_zone_metric.iloc[0]

    z_cols = st.columns(4)
    z_cols[0].metric("선택 구역 MAE", f"{z['MAE']:.3f}")
    z_cols[1].metric("선택 구역 MAPE", f"{z['MAPE(%)']:.2f}%")
    z_cols[2].metric("선택 구역 RMSE", f"{z['RMSE']:.3f}")
    z_cols[3].metric("선택 구역 R²", f"{z['R2']:.4f}")

st.dataframe(
    zone_metrics.sort_values("MAE").style.format({
        "MAE": "{:.3f}",
        "MAPE(%)": "{:.2f}",
        "RMSE": "{:.3f}",
        "R2": "{:.4f}"
    }),
    use_container_width=True
)


# ============================================================
# 9. 실제값 vs 예측값 그래프
# ============================================================

st.subheader(f"구역 {selected_zone} 실제값 vs 예측값")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=zone_df["timestamp"],
        y=zone_df["actual"],
        mode="lines",
        name="실제값",
        line=dict(width=3)
    )
)

for model_name in selected_models:
    pred_col = model_options[model_name]

    fig.add_trace(
        go.Scatter(
            x=zone_df["timestamp"],
            y=zone_df[pred_col],
            mode="lines",
            name=model_name,
            line=dict(width=2)
        )
    )

fig.update_layout(
    height=550,
    xaxis_title="시간",
    yaxis_title="인원 수",
    hovermode="x unified",
    legend_title="모델"
)

st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 10. 오차율 그래프
# ============================================================

st.subheader(f"구역 {selected_zone} 시간대별 오차율")

fig_error = go.Figure()

for model_name in selected_models:
    error_col = error_model_map[model_name]

    fig_error.add_trace(
        go.Scatter(
            x=zone_df["timestamp"],
            y=zone_df[error_col],
            mode="lines",
            name=f"{model_name} 오차율",
            line=dict(width=2)
        )
    )

fig_error.update_layout(
    height=450,
    xaxis_title="시간",
    yaxis_title="오차율 (%)",
    yaxis=dict(range=[0, 100]),
    hovermode="x unified",
    legend_title="모델"
)

st.plotly_chart(fig_error, use_container_width=True)


# ============================================================
# 11. 시간대별 절대 오차 그래프
# ============================================================

st.subheader(f"구역 {selected_zone} 시간대별 절대 오차")

abs_error_map = {
    "가중 앙상블": "ensemble_abs_error",
    "XGBoost": "xgb_abs_error",
    "RandomForest": "rf_abs_error"
}

fig_abs_error = go.Figure()

for model_name in selected_models:
    error_col = abs_error_map[model_name]

    fig_abs_error.add_trace(
        go.Scatter(
            x=zone_df["timestamp"],
            y=zone_df[error_col],
            mode="lines",
            name=f"{model_name} 절대 오차",
            line=dict(width=2)
        )
    )

fig_abs_error.update_layout(
    height=450,
    xaxis_title="시간",
    yaxis_title="절대 오차",
    hovermode="x unified",
    legend_title="모델"
)

st.plotly_chart(fig_abs_error, use_container_width=True)


# ============================================================
# 12. 전체 구역 히트맵
# ============================================================

if show_heatmap:
    heatmap_df = result.copy()
    heatmap_df["time"] = heatmap_df["timestamp"].dt.strftime("%H:%M")

    st.subheader("전체 구역 시간대별 앙상블 예측 히트맵")

    pivot_pred = heatmap_df.pivot_table(
        index="zone",
        columns="time",
        values="ensemble_pred",
        aggfunc="mean"
    )

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=pivot_pred.values,
            x=pivot_pred.columns,
            y=pivot_pred.index,
            colorbar=dict(title="예측 인원")
        )
    )

    fig_heatmap.update_layout(
        height=550,
        xaxis_title="시간",
        yaxis_title="구역"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("전체 구역 시간대별 실제 인원 히트맵")

    pivot_actual = heatmap_df.pivot_table(
        index="zone",
        columns="time",
        values="actual",
        aggfunc="mean"
    )

    fig_actual_heatmap = go.Figure(
        data=go.Heatmap(
            z=pivot_actual.values,
            x=pivot_actual.columns,
            y=pivot_actual.index,
            colorbar=dict(title="실제 인원")
        )
    )

    fig_actual_heatmap.update_layout(
        height=550,
        xaxis_title="시간",
        yaxis_title="구역"
    )

    st.plotly_chart(fig_actual_heatmap, use_container_width=True)

    st.subheader("전체 구역 시간대별 앙상블 오차율 히트맵")

    pivot_error = heatmap_df.pivot_table(
        index="zone",
        columns="time",
        values="ensemble_ape",
        aggfunc="mean"
    )

    fig_error_heatmap = go.Figure(
        data=go.Heatmap(
            z=pivot_error.values,
            x=pivot_error.columns,
            y=pivot_error.index,
            zmin=0,
            zmax=100,
            colorbar=dict(title="오차율 (%)")
        )
    )

    fig_error_heatmap.update_layout(
        height=550,
        xaxis_title="시간",
        yaxis_title="구역"
    )

    st.plotly_chart(fig_error_heatmap, use_container_width=True)


# ============================================================
# 13. 원본 데이터 테이블
# ============================================================

if show_raw_table:
    st.subheader("예측 결과 원본 데이터")

    st.dataframe(
        zone_df[
            [
                "timestamp",
                "zone",
                "actual",
                "xgb_pred",
                "rf_pred",
                "ensemble_pred",
                "xgb_abs_error",
                "rf_abs_error",
                "ensemble_abs_error",
                "xgb_ape",
                "rf_ape",
                "ensemble_ape"
            ]
        ],
        use_container_width=True
    )

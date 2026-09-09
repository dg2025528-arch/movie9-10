import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(page_title="영화 흥행 예측기", page_icon="🎬", layout="wide")

DAILY_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/"
    "data/kobis_daily.csv"
)
MOVIES_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/main/"
    "data/kobis_movies.csv"
)

PLOT_FLOOR = 1_000

# 원본 CSV의 열 이름이 안내된 한국어 이름 또는 영문 이름인 경우를 모두 처리합니다.
DAILY_RENAME = {
    "날짜": "date",
    "순위": "rank",
    "영화코드": "movieCd",
    "영화명": "movieNm_daily",
    "일관객": "daily_audi",
    "누적관객": "cumulative_audi",
    "스크린수": "screen_count",
    "상영횟수": "show_count",
    "audiCnt": "daily_audi",
    "audiAcc": "cumulative_audi",
    "scrnCnt": "screen_count",
    "showCnt": "show_count",
    "rank": "rank",
    "targetDt": "date",
    "movieNm": "movieNm_daily",
}

FEATURE_INFO = {
    "genre": ("장르", "categorical"),
    "nation": ("국가", "categorical"),
    "open_month": ("개봉 월", "numeric"),
    "peak": ("성수기 개봉 여부", "numeric"),
    "first_scrn": ("첫 관측일 스크린 수", "numeric"),
    "first_show": ("첫 관측일 상영 횟수", "numeric"),
    "first_week_audi": ("첫 주 관객 수", "numeric"),
    "days_in_top10": ("10위권 체류 일수", "numeric"),
    "daily_first_rank": ("10위권 첫 관측 순위", "numeric"),
    "daily_first_audi": ("10위권 첫 관측일 관객 수", "numeric"),
    "daily_peak_scrn": ("10위권 기간 최대 스크린 수", "numeric"),
    "daily_peak_show": ("10위권 기간 최대 상영 횟수", "numeric"),
}

DEFAULT_FEATURES = {
    "genre",
    "nation",
    "open_month",
    "peak",
    "first_scrn",
    "first_show",
    "first_week_audi",
    "days_in_top10",
}


def read_csv_from_url(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(
        io.BytesIO(response.content),
        encoding="utf-8",
        dtype={"movieCd": "string"},
    )


def clean_numeric(series: pd.Series) -> pd.Series:
    """쉼표 등이 들어간 숫자 열을 실수형으로 변환합니다."""
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


@st.cache_data(show_spinner=False)
def load_and_prepare_data():
    daily = read_csv_from_url(DAILY_URL).rename(columns=DAILY_RENAME)
    movies = read_csv_from_url(MOVIES_URL)

    if "movieCd" not in daily.columns or "movieCd" not in movies.columns:
        raise ValueError("두 CSV 모두에 movieCd 열이 있어야 합니다.")

    daily["movieCd"] = daily["movieCd"].astype("string").str.strip()
    movies["movieCd"] = movies["movieCd"].astype("string").str.strip()

    # 영화 정보 표의 영화는 중복 코드가 있더라도 한 편으로 취급합니다.
    movies = movies.drop_duplicates("movieCd", keep="first").copy()

    movie_numeric = [
        "first_scrn",
        "first_show",
        "peak",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]
    for column in movie_numeric:
        if column in movies.columns:
            movies[column] = clean_numeric(movies[column])

    daily_numeric = [
        "rank",
        "daily_audi",
        "cumulative_audi",
        "screen_count",
        "show_count",
    ]
    for column in daily_numeric:
        if column in daily.columns:
            daily[column] = clean_numeric(daily[column])

    if "date" in daily.columns:
        daily["date"] = pd.to_datetime(
            daily["date"].astype("string").str.replace(r"\.0$", "", regex=True),
            format="%Y%m%d",
            errors="coerce",
        )

    for column in ("openDt", "first_date"):
        if column in movies.columns:
            text = movies[column].astype("string").str.replace(r"\.0$", "", regex=True)
            movies[column] = pd.to_datetime(text, errors="coerce")

    if "openDt" in movies.columns:
        movies["open_month"] = movies["openDt"].dt.month.astype("float")
    else:
        movies["open_month"] = np.nan

    # 일별 자료를 영화별 변수로 집계합니다.
    if "date" in daily.columns:
        daily = daily.sort_values(["movieCd", "date", "rank"], na_position="last")
    else:
        daily = daily.sort_values(["movieCd", "rank"], na_position="last")

    aggregations = {}
    if "date" in daily.columns:
        aggregations["daily_first_date"] = ("date", "min")
        aggregations["daily_last_date"] = ("date", "max")
    if "rank" in daily.columns:
        aggregations["daily_first_rank"] = ("rank", "first")
    if "daily_audi" in daily.columns:
        aggregations["daily_first_audi"] = ("daily_audi", "first")
    if "screen_count" in daily.columns:
        aggregations["daily_peak_scrn"] = ("screen_count", "max")
    if "show_count" in daily.columns:
        aggregations["daily_peak_show"] = ("show_count", "max")

    if aggregations:
        daily_summary = (
            daily.groupby("movieCd", as_index=False)
            .agg(**aggregations)
        )
    else:
        daily_summary = daily[["movieCd"]].drop_duplicates()

    # movies를 왼쪽 표로 사용해 영화 정보 표의 모든 영화를 보존합니다.
    model_df = movies.merge(
        daily_summary,
        on="movieCd",
        how="left",
        validate="one_to_one",
    )

    for column in FEATURE_INFO:
        if column not in model_df.columns:
            model_df[column] = np.nan

    if "movieNm" not in model_df.columns:
        model_df["movieNm"] = model_df["movieCd"]

    model_df["genre"] = model_df["genre"].astype("string")
    model_df["nation"] = model_df["nation"].astype("string")
    model_df = model_df.sort_values("movieCd", kind="stable").reset_index(drop=True)

    # 영화코드 순서에서 각 10편마다 앞의 3편을 시험용으로 지정합니다.
    model_df["set"] = np.where(model_df.index % 10 < 3, "시험", "학습")

    daily_dates = daily["date"].dropna() if "date" in daily.columns else pd.Series(dtype="datetime64[ns]")
    movie_dates = []
    for column in ("openDt", "first_date"):
        if column in model_df.columns:
            movie_dates.append(model_df[column].dropna())

    all_dates = [series for series in [daily_dates, *movie_dates] if not series.empty]
    if all_dates:
        start_date = min(series.min() for series in all_dates)
        end_date = max(series.max() for series in all_dates)
    else:
        start_date = end_date = pd.NaT

    return model_df, start_date, end_date


def make_pipeline(selected_features):
    categorical = [
        feature
        for feature in selected_features
        if FEATURE_INFO[feature][1] == "categorical"
    ]
    numeric = [
        feature
        for feature in selected_features
        if FEATURE_INFO[feature][1] == "numeric"
    ]

    transformers = []

    if numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("number", numeric_pipeline, numeric))

    if categorical:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="정보 없음"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(("category", categorical_pipeline, categorical))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def fmt_date(value):
    if pd.isna(value):
        return "확인 불가"
    return pd.Timestamp(value).strftime("%Y-%m-%d")


st.title("🎬 영화 흥행 예측기")
st.caption(
    "박스오피스 일별 자료와 영화 정보 표를 movieCd로 합쳐 "
    "다중 선형 회귀로 총 관객 수를 예측합니다."
)

try:
    with st.spinner("KOBIS 자료를 불러오는 중입니다..."):
        df, period_start, period_end = load_and_prepare_data()
except Exception as error:
    st.error(f"자료를 불러오지 못했습니다: {error}")
    st.stop()

st.sidebar.header("예측 변수 선택")
st.sidebar.caption("영화 정보 표의 모든 영화가 분할 대상에 포함됩니다.")

selected_features = []
for feature, (label, _) in FEATURE_INFO.items():
    if st.sidebar.checkbox(
        label,
        value=feature in DEFAULT_FEATURES,
        key=f"feature_{feature}",
    ):
        selected_features.append(feature)

if not selected_features:
    st.warning("왼쪽에서 예측에 사용할 변수를 한 개 이상 선택해 주세요.")
    st.stop()

if df["total_audi"].isna().any():
    missing_target = int(df["total_audi"].isna().sum())
    st.error(
        f"영화 정보 표의 영화 {missing_target:,}편에 total_audi가 없어 "
        "모든 영화를 학습·평가에 사용할 수 없습니다."
    )
    st.stop()

train_df = df.loc[df["set"] == "학습"].copy()
test_df = df.loc[df["set"] == "시험"].copy()

if train_df.empty or test_df.empty:
    st.error("학습용 또는 시험용 영화가 없습니다.")
    st.stop()

pipeline = make_pipeline(selected_features)

try:
    pipeline.fit(train_df[selected_features], train_df["total_audi"])
    raw_prediction = pipeline.predict(test_df[selected_features])
except Exception as error:
    st.error(f"모델을 학습하지 못했습니다: {error}")
    st.stop()

# 회귀값은 음수가 될 수 있으므로 관객 수 표시와 평가에서는 0으로 제한합니다.
prediction = np.maximum(raw_prediction, 0)
actual = test_df["total_audi"].to_numpy(dtype=float)

r2 = r2_score(actual, prediction) if len(test_df) >= 2 else np.nan
mae = mean_absolute_error(actual, prediction)
rmse = mean_squared_error(actual, prediction) ** 0.5

test_df["actual_total_audi"] = actual
test_df["predicted_total_audi"] = prediction
test_df["error"] = prediction - actual
test_df["absolute_error"] = np.abs(test_df["error"])
test_df["absolute_percentage_error"] = np.where(
    actual > 0,
    test_df["absolute_error"] / actual * 100,
    np.nan,
)

below_floor = test_df["predicted_total_audi"] < PLOT_FLOOR
below_floor_count = int(below_floor.sum())

info1, info2, info3, info4 = st.columns(4)
info1.metric("학습 영화", f"{len(train_df):,}편")
info2.metric("시험 영화", f"{len(test_df):,}편")
info3.metric("전체 영화", f"{len(df):,}편")
info4.metric("기준 기간", f"{fmt_date(period_start)} ~ {fmt_date(period_end)}")

st.caption(
    "분할 규칙: 영화별 표를 movieCd 오름차순으로 정렬한 뒤 "
    "매 10편의 앞 3편은 시험용, 나머지 7편은 학습용으로 사용했습니다."
)

st.subheader("시험용 영화 예측 점수")
score1, score2, score3, score4 = st.columns(4)
score1.metric("결정계수 R²", "계산 불가" if np.isnan(r2) else f"{r2:.3f}")
score2.metric("평균 절대 오차 MAE", f"{mae:,.0f}명")
score3.metric("평균 제곱근 오차 RMSE", f"{rmse:,.0f}명")
score4.metric("1,000명 미만 예측", f"{below_floor_count:,}편")

st.caption(
    "R²는 1에 가까울수록 좋습니다. MAE와 RMSE는 실제 총 관객 수와 "
    "예측 총 관객 수가 평균적으로 얼마나 빗나갔는지를 나타냅니다."
)

st.subheader("실제 총 관객 수와 예측 총 관객 수")

plot_df = test_df.copy()
plot_df["plot_actual"] = np.maximum(plot_df["actual_total_audi"], 1)
plot_df["plot_prediction"] = np.maximum(
    plot_df["predicted_total_audi"],
    PLOT_FLOOR,
)
plot_df["floor_label"] = np.where(
    plot_df["predicted_total_audi"] < PLOT_FLOOR,
    "1,000명 미만 — 그래프 바닥에 표시",
    "일반 예측",
)

positive_values = np.concatenate(
    [
        plot_df["plot_actual"].to_numpy(),
        plot_df["plot_prediction"].to_numpy(),
        np.array([PLOT_FLOOR], dtype=float),
    ]
)
axis_min = max(1, float(np.nanmin(positive_values)) * 0.7)
axis_max = max(PLOT_FLOOR * 1.2, float(np.nanmax(positive_values)) * 1.3)

fig = go.Figure()

normal = plot_df["predicted_total_audi"] >= PLOT_FLOOR
fig.add_trace(
    go.Scatter(
        x=plot_df.loc[normal, "plot_actual"],
        y=plot_df.loc[normal, "plot_prediction"],
        mode="markers",
        name="예측 영화",
        marker=dict(size=9, color="#2E86DE", opacity=0.75),
        customdata=np.column_stack(
            [
                plot_df.loc[normal, "movieNm"].astype(str),
                plot_df.loc[normal, "movieCd"].astype(str),
                plot_df.loc[normal, "actual_total_audi"],
                plot_df.loc[normal, "predicted_total_audi"],
                plot_df.loc[normal, "error"],
            ]
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "영화코드: %{customdata[1]}<br>"
            "실제: %{customdata[2]:,.0f}명<br>"
            "예측: %{customdata[3]:,.0f}명<br>"
            "오차: %{customdata[4]:+,.0f}명"
            "<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=plot_df.loc[below_floor, "plot_actual"],
        y=plot_df.loc[below_floor, "plot_prediction"],
        mode="markers",
        name=f"1,000명 미만 예측 ({below_floor_count}편)",
        marker=dict(
            size=10,
            color="#E74C3C",
            symbol="triangle-up",
            opacity=0.9,
        ),
        customdata=np.column_stack(
            [
                plot_df.loc[below_floor, "movieNm"].astype(str),
                plot_df.loc[below_floor, "movieCd"].astype(str),
                plot_df.loc[below_floor, "actual_total_audi"],
                plot_df.loc[below_floor, "predicted_total_audi"],
                plot_df.loc[below_floor, "error"],
            ]
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "영화코드: %{customdata[1]}<br>"
            "실제: %{customdata[2]:,.0f}명<br>"
            "예측: %{customdata[3]:,.0f}명<br>"
            "오차: %{customdata[4]:+,.0f}명<br>"
            "※ 그래프에서는 1,000명 위치에 표시"
            "<extra></extra>"
        ),
    )
)

fig.add_trace(
    go.Scatter(
        x=[axis_min, axis_max],
        y=[axis_min, axis_max],
        mode="lines",
        name="실제 = 예측",
        line=dict(color="#444444", width=2, dash="dash"),
        hoverinfo="skip",
    )
)

fig.update_layout(
    height=650,
    template="plotly_white",
    xaxis_title="실제 총 관객 수",
    yaxis_title="예측 총 관객 수",
    legend_title_text="",
    hovermode="closest",
    margin=dict(l=30, r=20, t=30, b=30),
)
fig.update_xaxes(type="log", range=[np.log10(axis_min), np.log10(axis_max)])
fig.update_yaxes(type="log", range=[np.log10(axis_min), np.log10(axis_max)])

st.plotly_chart(fig, use_container_width=True)
st.caption(
    f"두 축은 로그 눈금입니다. 예측값이 {PLOT_FLOOR:,}명보다 작은 "
    f"{below_floor_count:,}편은 실제 예측값과 관계없이 그래프의 "
    f"{PLOT_FLOOR:,}명 위치에 붙여 표시했습니다."
)

st.subheader("시험용 영화별 실제값·예측값·오차")

result_columns = [
    "movieCd",
    "movieNm",
    "actual_total_audi",
    "predicted_total_audi",
    "error",
    "absolute_error",
    "absolute_percentage_error",
]
result_table = (
    test_df[result_columns]
    .sort_values("movieCd", kind="stable")
    .rename(
        columns={
            "movieCd": "영화코드",
            "movieNm": "영화명",
            "actual_total_audi": "실제 총 관객 수",
            "predicted_total_audi": "예측 총 관객 수",
            "error": "오차(예측-실제)",
            "absolute_error": "절대 오차",
            "absolute_percentage_error": "절대 백분율 오차(%)",
        }
    )
)

st.dataframe(
    result_table.style.format(
        {
            "실제 총 관객 수": "{:,.0f}",
            "예측 총 관객 수": "{:,.0f}",
            "오차(예측-실제)": "{:+,.0f}",
            "절대 오차": "{:,.0f}",
            "절대 백분율 오차(%)": "{:,.1f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

csv_data = result_table.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "시험 결과 CSV 내려받기",
    data=csv_data,
    file_name="movie_boxoffice_predictions.csv",
    mime="text/csv",
)

with st.expander("선택한 변수와 데이터 결합 방식"):
    labels = [FEATURE_INFO[feature][0] for feature in selected_features]
    st.write("**선택 변수:** " + ", ".join(labels))
    st.write(
        "kobis_movies.csv를 기준으로 kobis_daily.csv의 영화별 집계값을 "
        "movieCd로 왼쪽 결합하여 영화 정보 표의 모든 영화를 보존했습니다."
    )
    st.write(
        "범주형 변수는 원-핫 인코딩하고, 수치형 변수의 결측값은 학습 자료의 "
        "중앙값으로 대체한 뒤 다중 선형 회귀를 학습했습니다."
    )

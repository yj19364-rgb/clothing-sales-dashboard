
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ======================================
# 1. 페이지 설정 및 데이터 불러오기
# ======================================

st.set_page_config(
    page_title="의류 판매량 분석",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "model_detail.csv"
MODEL_PATH = BASE_DIR / "clothing_sales_model.joblib"

st.title("📊 기온 기반 의류 판매량 분석")
st.write("실제 의류 판매 데이터와 머신러닝 모델을 활용한 분석 대시보드입니다.")

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(data["Date"])
    return data

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "데이터 또는 모델 파일을 찾을 수 없습니다. "
        "app.py, model_detail.csv, clothing_sales_model.joblib이 "
        "같은 폴더에 있는지 확인해 주세요."
    )
    st.stop()

df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

required_columns = [
    "Date", "Units Sold",
    "평균기온", "최고기온", "최저기온",
    "월", "요일번호", "Store ID", "Product ID",
    "Region", "Price", "Discount", "Promotion",
    "Inventory Level", "Competitor Pricing",
    "Seasonality", "Weather Condition"
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    st.error(f"데이터에 필요한 컬럼이 없습니다: {missing_columns}")
    st.stop()

# ======================================
# 2. 사이드바
# ======================================

st.sidebar.header("📌 분석 설정")

start_date = st.sidebar.date_input(
    "시작 날짜",
    value=df["Date"].min().date(),
    min_value=df["Date"].min().date(),
    max_value=df["Date"].max().date()
)

end_date = st.sidebar.date_input(
    "종료 날짜",
    value=df["Date"].max().date(),
    min_value=df["Date"].min().date(),
    max_value=df["Date"].max().date()
)

if start_date > end_date:
    st.error("시작 날짜는 종료 날짜보다 늦을 수 없습니다.")
    st.stop()

filtered_df = df[
    (df["Date"].dt.date >= start_date) &
    (df["Date"].dt.date <= end_date)
].copy()

# ======================================
# 3. 핵심 요약 지표
# ======================================

daily_sales = (
    filtered_df.groupby("Date", as_index=False)
    .agg({
        "Units Sold": "sum",
        "평균기온": "mean",
        "최고기온": "mean",
        "최저기온": "mean"
    })
)

if daily_sales.empty:
    st.warning("선택한 기간에 데이터가 없습니다.")
    st.stop()

total_sales = int(filtered_df["Units Sold"].sum())
average_daily_sales = daily_sales["Units Sold"].mean()
average_temperature = daily_sales["평균기온"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("분석 날짜 수", f"{len(daily_sales):,}일")
col2.metric("총 의류 판매량", f"{total_sales:,}개")
col3.metric("일평균 판매량", f"{average_daily_sales:,.1f}개")
col4.metric("평균 기온", f"{average_temperature:.1f}°C")

st.divider()

# ======================================
# 4. 날짜별 판매량 그래프
# ======================================

st.subheader("📈 날짜별 전체 의류 판매량")

sales_fig = px.line(
    daily_sales,
    x="Date",
    y="Units Sold",
    labels={"Date": "날짜", "Units Sold": "판매량 (개)"},
    title=None
)
sales_fig.update_xaxes(
    tickangle=0,
    tickformat="%Y-%m",
    dtick="M1",
    automargin=True,
    title_standoff=12
)
sales_fig.update_yaxes(tickangle=0, automargin=True)
sales_fig.update_layout(
    height=420,
    margin=dict(l=20, r=20, t=20, b=50),
    hovermode="x unified"
)
st.plotly_chart(sales_fig, use_container_width=True)

# ======================================
# 5. 기온과 판매량의 관계
# ======================================

st.subheader("🌡️ 기온과 의류 판매량의 관계")

left, right = st.columns(2)

with left:
    st.write("**평균 기온별 일일 판매량**")

    temp_fig = px.scatter(
        daily_sales,
        x="평균기온",
        y="Units Sold",
        labels={"평균기온": "평균 기온 (°C)", "Units Sold": "일일 판매량 (개)"},
        hover_data={"Date": True, "최고기온": True, "최저기온": True}
    )
    temp_fig.update_xaxes(tickangle=0, automargin=True)
    temp_fig.update_yaxes(tickangle=0, automargin=True)
    temp_fig.update_layout(height=380, margin=dict(l=20, r=15, t=20, b=45))
    st.plotly_chart(temp_fig, use_container_width=True)

with right:
    st.write("**기온 구간별 평균 판매량**")

    bins = [-float("inf"), 0, 10, 20, 30, float("inf")]
    labels = ["0°C 미만", "0~10°C", "10~20°C", "20~30°C", "30°C 이상"]

    temp_df = daily_sales.copy()
    temp_df["기온 구간"] = pd.cut(
        temp_df["평균기온"],
        bins=bins,
        labels=labels,
        right=False
    )

    temp_summary = (
        temp_df.groupby("기온 구간", observed=False)["Units Sold"]
        .mean()
        .dropna()
        .reset_index()
    )

    temp_fig_bar = px.bar(
        temp_summary,
        x="기온 구간",
        y="Units Sold",
        labels={"기온 구간": "평균 기온 구간", "Units Sold": "평균 일일 판매량 (개)"}
    )
    temp_fig_bar.update_xaxes(tickangle=0, automargin=True)
    temp_fig_bar.update_yaxes(tickangle=0, automargin=True)
    temp_fig_bar.update_layout(height=380, margin=dict(l=20, r=15, t=20, b=55))
    st.plotly_chart(temp_fig_bar, use_container_width=True)

st.caption(
    "기온은 한국 대표 도시 9곳의 일별 평균을 사용했습니다. "
    "기온과 판매량의 관계가 곧 인과관계를 의미하지는 않습니다."
)

# ======================================
# 6. 월별 판매량 그래프
# ======================================

st.subheader("📅 월별 평균 일일 의류 판매량")

daily_sales["월"] = daily_sales["Date"].dt.month

monthly_sales = (
    daily_sales.groupby("월", as_index=False)["Units Sold"]
    .mean()
)

monthly_sales["월"] = monthly_sales["월"].astype(str) + "월"

monthly_fig = px.bar(
    monthly_sales,
    x="월",
    y="Units Sold",
    labels={"월": "월", "Units Sold": "평균 일일 판매량 (개)"}
)
monthly_fig.update_xaxes(tickangle=0, automargin=True)
monthly_fig.update_yaxes(tickangle=0, automargin=True)
monthly_fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=50))
st.plotly_chart(monthly_fig, use_container_width=True)

st.divider()

# ======================================
# 7. 머신러닝 판매량 예측
# ======================================

st.header("🤖 의류 판매량 예측")
st.write(
    "날짜, 기온, 상품 및 매장 조건을 입력하면 "
    "학습된 모델이 해당 상품·매장의 예상 판매량을 계산합니다."
)

st.info(
    "예측 대상은 선택한 상품과 매장의 판매량입니다. "
    "전체 매장의 일일 총판매량 예측과는 다릅니다."
)

# 원래 모델에 사용한 입력 컬럼 순서를 유지
feature_columns = [
    "평균기온", "최고기온", "최저기온", "월", "요일번호",
    "Store ID", "Product ID", "Region", "Price", "Discount",
    "Promotion", "Inventory Level", "Competitor Pricing",
    "Seasonality", "Weather Condition"
]

def options_for(column):
    return sorted(df[column].dropna().unique().tolist(), key=str)

with st.form("prediction_form"):
    st.subheader("예측 조건 입력")

    predict_date = st.date_input(
        "예측 날짜",
        value=df["Date"].max().date()
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        avg_temp = st.number_input(
            "평균 기온 (°C)",
            value=float(df["평균기온"].median()),
            step=1.0
        )

        max_temp = st.number_input(
            "최고 기온 (°C)",
            value=float(df["최고기온"].median()),
            step=1.0
        )

        min_temp = st.number_input(
            "최저 기온 (°C)",
            value=float(df["최저기온"].median()),
            step=1.0
        )

    with c2:
        store_id = st.selectbox("매장 ID", options_for("Store ID"))
        product_id = st.selectbox("상품 ID", options_for("Product ID"))
        region = st.selectbox("지역", options_for("Region"))
        seasonality = st.selectbox(
            "계절",
            options_for("Seasonality")
        )
        weather = st.selectbox(
            "날씨 상태",
            options_for("Weather Condition")
        )

    with c3:
        price = st.number_input(
            "상품 가격",
            min_value=0.0,
            value=float(df["Price"].median()),
            step=1.0
        )

        discount = st.number_input(
            "할인율",
            min_value=0.0,
            max_value=100.0,
            value=float(df["Discount"].median()),
            step=1.0
        )

        promotion = st.selectbox(
            "프로모션 여부",
            options_for("Promotion")
        )

        inventory = st.number_input(
            "재고 수준",
            min_value=0,
            value=int(df["Inventory Level"].median()),
            step=1
        )

        competitor_price = st.number_input(
            "경쟁사 가격",
            min_value=0.0,
            value=float(df["Competitor Pricing"].median()),
            step=1.0
        )

    submitted = st.form_submit_button("판매량 예측하기")

if submitted:
    if min_temp > avg_temp or avg_temp > max_temp:
        st.error("기온은 최저 기온 ≤ 평균 기온 ≤ 최고 기온이어야 합니다.")
    else:
        input_data = pd.DataFrame([{
            "평균기온": avg_temp,
            "최고기온": max_temp,
            "최저기온": min_temp,
            "월": predict_date.month,
            "요일번호": predict_date.weekday(),
            "Store ID": store_id,
            "Product ID": product_id,
            "Region": region,
            "Price": price,
            "Discount": discount,
            "Promotion": promotion,
            "Inventory Level": inventory,
            "Competitor Pricing": competitor_price,
            "Seasonality": seasonality,
            "Weather Condition": weather
        }], columns=feature_columns)

        try:
            prediction = float(model.predict(input_data)[0])
            prediction = max(0, prediction)

            st.success("판매량 예측이 완료되었습니다.")
            st.metric("예상 판매량", f"{prediction:.0f}개")
            st.caption(
                "모델이 학습한 과거 데이터의 패턴을 바탕으로 계산한 추정치입니다. "
                "실제 판매량을 보장하지 않습니다."
            )

        except Exception as e:
            st.error("예측 중 오류가 발생했습니다.")
            st.code(str(e))

st.divider()
st.caption("의류 판매량 분석 프로젝트 | Streamlit 대시보드")

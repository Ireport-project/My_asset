"""자산 수익률 추적 웹 앱.

KDB생명, 교보생명의 납입금/환급금을 날짜별로 기록하고
수익을 꺾은선 그래프로 시각화한다.

실행 방법:
    streamlit run main.py
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Tuple
from zoneinfo import ZoneInfo

import firebase_admin
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from firebase_admin import credentials
from firebase_admin import firestore
from plotly.subplots import make_subplots
from yaml.loader import SafeLoader

_APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = _APP_DIR / "config.yaml"

ASSETS = ["KDB생명", "교보생명"]
COLUMNS = ["date", "asset", "payment", "refund"]
COLLECTION_NAME = "asset_records"

# st.date_input 기본은 초기값 기준 ±10년이라 과거 기록(2000년대 등)이 막힐 수 있음 → 범위 명시
EDITABLE_DATE_MIN = date(2000, 1, 1)
EDITABLE_DATE_MAX = date(2100, 12, 31)
KOREA_TZ = ZoneInfo("Asia/Seoul")
CHART_VIEW_MODES = ["합산", "KDB생명", "교보생명"]
CHART_PERIODS = [
    "최근 1달",
    "최근 6개월",
    "최근 1년",
    "최근 5년",
    "MAX(전체기간)",
]
CHART_PERIODS_WITH_LABELS = {"최근 1달", "최근 6개월", "최근 1년"}
CHART_DATE_FORMAT = "%Y-%m-%d"

if not firebase_admin._apps:
    firebase_admin.initialize_app(
        credentials.Certificate(dict(st.secrets["firebase"]))
    )
db = firestore.client()

AMOUNT_FIELD_KEYS = (
    "kdb_payment_input",
    "kdb_refund_input",
    "kyobo_payment_input",
    "kyobo_refund_input",
)


def korea_today() -> date:
    """한국 시간(Asia/Seoul) 기준 오늘 날짜."""
    return datetime.now(KOREA_TZ).date()


def _to_kst_date(value: object) -> pd.Timestamp:
    """Firestore/로컬 환경 차이를 줄이기 위해 KST 기준 날짜(00:00)로 통일."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(KOREA_TZ).tz_localize(None)
    return ts.normalize()


def _parse_record_date(value: object) -> pd.Timestamp:
    """Firestore에서 읽은 날짜 값을 KST 기준 Timestamp로 변환한다."""
    if value is None:
        return pd.NaT
    if isinstance(value, datetime):
        return _to_kst_date(value)
    if isinstance(value, date) and not isinstance(value, datetime):
        return _to_kst_date(value.isoformat())
    return _to_kst_date(value)


def format_default_amount(value: int) -> str:
    """입력 필드 기본값용 금액 문자열. 0이면 빈 칸."""
    return f"{value:,}" if value > 0 else ""


def get_latest_payment_by_asset(df: pd.DataFrame) -> dict[str, int]:
    """자산별 가장 최근 날짜 기록의 누적 납입금을 반환한다."""
    latest: dict[str, int] = {}
    if df.empty:
        return latest
    for asset in ASSETS:
        asset_df = df[df["asset"] == asset]
        if asset_df.empty:
            continue
        latest_date = asset_df["date"].max()
        row = asset_df[asset_df["date"] == latest_date].iloc[-1]
        latest[asset] = int(row["payment"])
    return latest


def parse_amount(value: object) -> int:
    """콤마가 포함된 문자열을 정수 금액으로 변환한다. 빈 값/잘못된 값은 0."""
    if value is None:
        return 0
    text = str(value).replace(",", "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def format_amount_field(state_key: str) -> None:
    """text_input의 입력값을 천 단위 콤마(#,##0) 형태로 정규화하는 콜백."""
    raw = st.session_state.get(state_key, "")
    if not isinstance(raw, str):
        return
    cleaned = raw.replace(",", "").strip()
    if cleaned == "":
        st.session_state[state_key] = ""
        return
    try:
        num = int(float(cleaned))
        st.session_state[state_key] = f"{num:,}"
    except ValueError:
        st.session_state[state_key] = ""


def _delete_all_collection_docs() -> None:
    """컬렉션의 모든 문서를 배치 단위로 삭제한다."""
    coll = db.collection(COLLECTION_NAME)
    batch_size = 500
    while True:
        docs = list(coll.limit(batch_size).stream())
        if not docs:
            break
        batch = db.batch()
        for doc in docs:
            batch.delete(doc.reference)
        batch.commit()


def load_data() -> pd.DataFrame:
    """Firestore에서 자산 기록을 불러온다. 없으면 빈 DataFrame을 만든다."""
    rows: list[dict[str, object]] = []
    for doc in db.collection(COLLECTION_NAME).stream():
        d = doc.to_dict()
        if not d:
            continue
        rows.append(
            {
                "date": _parse_record_date(d.get("date")),
                "asset": d.get("asset", ""),
                "payment": d.get("payment", 0),
                "refund": d.get("refund", 0),
            }
        )
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        df = pd.DataFrame(columns=COLUMNS)
        df["date"] = pd.to_datetime(df["date"])
    else:
        df = df.dropna(subset=["date"])
        df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    for col in ("payment", "refund"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def save_data(df: pd.DataFrame) -> None:
    """컬렉션을 비운 뒤 DataFrame 내용을 배치로 다시 저장한다."""
    _delete_all_collection_docs()
    coll = db.collection(COLLECTION_NAME)
    df_to_save = df.copy()
    df_to_save["date"] = pd.to_datetime(df_to_save["date"]).dt.strftime("%Y-%m-%d")
    records = df_to_save[COLUMNS].to_dict("records")
    batch_size = 500
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        batch = db.batch()
        for rec in chunk:
            ref = coll.document()
            batch.set(
                ref,
                {
                    "date": rec["date"],
                    "asset": rec["asset"],
                    "payment": float(rec["payment"]),
                    "refund": float(rec["refund"]),
                },
            )
        batch.commit()


def add_record(
    record_date: date, asset: str, payment: float, refund: float
) -> None:
    """새로운 기록을 추가한다. 동일 (날짜, 자산) 키가 있으면 덮어쓴다."""
    coll = db.collection(COLLECTION_NAME)
    dstr = record_date.isoformat()
    q = (
        coll.where("date", "==", dstr)
        .where("asset", "==", asset)
        .limit(1)
    )
    existing = list(q.stream())
    payload = {
        "date": dstr,
        "asset": asset,
        "payment": float(payment),
        "refund": float(refund),
    }
    if existing:
        existing[0].reference.update(
            {"payment": payload["payment"], "refund": payload["refund"]}
        )
    else:
        coll.add(payload)


def build_summary(df: pd.DataFrame, selected_assets: list[str]) -> pd.DataFrame:
    """선택한 자산을 기준으로 날짜별 누적 납입금/누적 환급금/수익을 계산한다.

    - 납입금, 환급금 모두 그 시점까지의 누적 금액으로 입력받았다고 가정한다.
    - 같은 (날짜, 자산) 키에 여러 기록이 있으면 가장 최신 값을 사용한다.
    - 여러 자산을 합산할 때는 각 자산별 시계열을 forward-fill 한 뒤 더한다.
    """
    if df.empty or not selected_assets:
        return pd.DataFrame(
            columns=["date", "cum_payment", "current_refund", "profit", "rate"]
        )

    filtered = df[df["asset"].isin(selected_assets)].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=["date", "cum_payment", "current_refund", "profit", "rate"]
        )

    filtered = filtered.sort_values(["asset", "date"])
    all_dates = sorted(filtered["date"].unique())

    cum_payment_total = pd.Series(0.0, index=all_dates)
    current_refund_total = pd.Series(0.0, index=all_dates)

    for asset in selected_assets:
        asset_df = filtered[filtered["asset"] == asset].sort_values("date")
        if asset_df.empty:
            continue
        asset_df = asset_df.groupby("date", as_index=True).agg(
            payment=("payment", "last"),
            refund=("refund", "last"),
        )
        cum_payment = asset_df["payment"].reindex(all_dates).ffill().fillna(0)
        cur_refund = asset_df["refund"].reindex(all_dates).ffill().fillna(0)

        cum_payment_total = cum_payment_total.add(cum_payment, fill_value=0)
        current_refund_total = current_refund_total.add(cur_refund, fill_value=0)

    summary = pd.DataFrame(
        {
            "date": all_dates,
            "cum_payment": cum_payment_total.values,
            "current_refund": current_refund_total.values,
        }
    )
    summary["profit"] = summary["current_refund"] - summary["cum_payment"]
    summary["rate"] = summary.apply(
        lambda r: (r["profit"] / r["cum_payment"] * 100) if r["cum_payment"] else 0.0,
        axis=1,
    )
    return summary


def filter_summary_by_period(summary: pd.DataFrame, period_label: str) -> pd.DataFrame:
    """한국 시간 기준 오늘을 끝점으로 선택한 기간의 summary만 남긴다."""
    if summary.empty or period_label == "MAX(전체기간)":
        return summary

    end = pd.Timestamp(korea_today())
    offsets = {
        "최근 1달": pd.DateOffset(months=1),
        "최근 6개월": pd.DateOffset(months=6),
        "최근 1년": pd.DateOffset(years=1),
        "최근 5년": pd.DateOffset(years=5),
    }
    start = end - offsets[period_label]
    dates = pd.to_datetime(summary["date"])
    filtered = summary.loc[dates >= start].copy()
    return filtered.reset_index(drop=True)


def _summary_timeline(summary: pd.DataFrame) -> pd.DataFrame:
    """수익 보간용 날짜별 누적 납입금·환급금 시계열."""
    timeline = summary.copy()
    timeline["date"] = pd.to_datetime(timeline["date"], errors="coerce").map(_to_kst_date)
    timeline = timeline.dropna(subset=["date"]).sort_values("date")
    timeline = timeline.drop_duplicates(subset=["date"], keep="last")
    for col in ("cum_payment", "current_refund"):
        timeline[col] = pd.to_numeric(timeline[col], errors="coerce")
    timeline = timeline.dropna(subset=["cum_payment", "current_refund"])
    return timeline.reset_index(drop=True)


def _interpolate_amounts(
    timeline: pd.DataFrame, target: pd.Timestamp
) -> Optional[tuple[float, float]]:
    """기준일의 (누적 납입금, 누적 환급금) 보간값."""
    if timeline.empty:
        return None

    target = _to_kst_date(target)
    if len(timeline) == 1:
        row = timeline.iloc[0]
        return float(row["cum_payment"]), float(row["current_refund"])

    x = timeline["date"].astype(np.int64).to_numpy()
    payment = timeline["cum_payment"].to_numpy(dtype=float)
    refund = timeline["current_refund"].to_numpy(dtype=float)
    target_x = target.value
    return (
        float(np.interp(target_x, x, payment)),
        float(np.interp(target_x, x, refund)),
    )


def _value_series(summary: pd.DataFrame, column: str) -> pd.Series:
    """summary에서 날짜별 시계열 (동일 날짜는 마지막 값)."""
    s = summary.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    s = s.dropna(subset=["date"])
    s["date"] = s["date"].map(_to_kst_date)
    s = s.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return s.set_index("date")[column].astype(float)


def _profit_series(summary: pd.DataFrame) -> pd.Series:
    """날짜별 수익 시계열 (동일 날짜는 마지막 값)."""
    return _value_series(summary, "profit")


def interpolated_at(series: pd.Series, target: pd.Timestamp) -> Optional[float]:
    """시간축 선형 보간으로 기준일 값을 추정한다."""
    if series.empty:
        return None

    target = _to_kst_date(target)
    normalized = series.copy()
    normalized.index = pd.Index(_to_kst_date(d) for d in normalized.index)
    sorted_series = normalized.groupby(level=0).last().sort_index()
    if len(sorted_series) == 1:
        return float(sorted_series.iloc[0])

    x = sorted_series.index.astype(np.int64).to_numpy()
    y = sorted_series.to_numpy()
    return float(np.interp(target.value, x, y))


def profit_at_summary(summary: pd.DataFrame, target: pd.Timestamp) -> Optional[float]:
    """기준일 누적 환급금·납입금 보간값으로 수익을 계산한다."""
    amounts = _interpolate_amounts(_summary_timeline(summary), target)
    if amounts is None:
        return None
    payment, refund = amounts
    return refund - payment


def interpolated_profit_at(profit_series: pd.Series, target: pd.Timestamp) -> Optional[float]:
    """시간축 선형 보간으로 기준일 수익을 추정한다."""
    return interpolated_at(profit_series, target)


def compute_profit_changes(summary: pd.DataFrame) -> dict[str, Optional[float]]:
    """최신 기록일 기준 전일/전월/1년전/년초 수익 증감액 (보간값)."""
    if summary.empty:
        return {
            "전일 대비": None,
            "전월 대비": None,
            "1년전 대비": None,
            "년초 대비": None,
        }

    timeline = _summary_timeline(summary)
    if timeline.empty:
        return {
            "전일 대비": None,
            "전월 대비": None,
            "1년전 대비": None,
            "년초 대비": None,
        }

    eval_date = _to_kst_date(timeline["date"].iloc[-1])
    current_profit = profit_at_summary(summary, eval_date)
    if current_profit is None:
        return {
            "전일 대비": None,
            "전월 대비": None,
            "1년전 대비": None,
            "년초 대비": None,
        }

    def change_since(target: pd.Timestamp) -> Optional[float]:
        past_profit = profit_at_summary(summary, target)
        if past_profit is None:
            return None
        return current_profit - past_profit

    return {
        "전일 대비": change_since(eval_date - pd.Timedelta(days=1)),
        "전월 대비": change_since(eval_date - pd.DateOffset(months=1)),
        "1년전 대비": change_since(eval_date - pd.DateOffset(years=1)),
        "년초 대비": change_since(pd.Timestamp(eval_date.year, 1, 1)),
    }


def format_signed_won(value: float) -> str:
    """부호가 있는 원 단위 문자열."""
    return f"{int(round(value)):+,} 원"


def format_compact_amount(value: float) -> str:
    """차트 레이블용 축약 금액 (예: 114,700,000 → 115M)."""
    amount = float(value)
    sign = "-" if amount < 0 else ""
    n = abs(amount)
    if n >= 1_000_000_000:
        return f"{sign}{round(n / 1_000_000_000)}B"
    if n >= 1_000_000:
        return f"{sign}{round(n / 1_000_000)}M"
    if n >= 1_000:
        return f"{sign}{round(n / 1_000)}K"
    return f"{sign}{int(round(n))}"


def format_profit_change(change: float, cum_payment: float) -> str:
    """수익 증감액과 누적 납입금 대비 비율."""
    text = format_signed_won(change)
    if cum_payment > 0:
        rate = change / cum_payment * 100
        text += f" ({rate:+.2f}%)"
    return text


def build_yearly_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """각 연도 12월 31일 기준 누적 납입금·환급금·수익률 (보간값)."""
    if summary.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "cum_payment",
                "current_refund",
                "profit",
                "rate",
                "profit_yoy",
                "rate_yoy",
            ]
        )

    payment_series = _value_series(summary, "cum_payment")
    refund_series = _value_series(summary, "current_refund")
    profit_series = _profit_series(summary)
    first_year = int(payment_series.index.min().year)
    last_year = int(payment_series.index.max().year)

    rows: list[dict[str, object]] = []
    for year in range(first_year, last_year + 1):
        target = pd.Timestamp(year, 12, 31)
        payment = interpolated_at(payment_series, target) or 0.0
        refund = interpolated_at(refund_series, target) or 0.0
        profit = interpolated_at(profit_series, target) or 0.0
        rate = (profit / payment * 100) if payment > 0 else 0.0
        rows.append(
            {
                "year": year,
                "cum_payment": payment,
                "current_refund": refund,
                "profit": profit,
                "rate": rate,
            }
        )

    yearly = pd.DataFrame(rows)
    yearly["profit_yoy"] = yearly["profit"].diff()
    yearly["rate_yoy"] = yearly["rate"].diff()
    return yearly


def payment_duration_months(summary: pd.DataFrame) -> int:
    """첫 기록~최신 기록 사이 개월 수 (최소 1)."""
    s = summary.copy()
    s["date"] = pd.to_datetime(s["date"])
    if s.empty:
        return 0
    first = s["date"].min()
    last = s["date"].max()
    months = (last.year - first.year) * 12 + (last.month - first.month)
    if last.day < first.day:
        months -= 1
    return max(months, 1)


def format_payment_duration(summary: pd.DataFrame) -> str:
    """첫 기록~최신 기록 사이 납입 기간."""
    months = payment_duration_months(summary)
    if months <= 0:
        return "—"
    years, rem_months = divmod(months, 12)
    parts: list[str] = []
    if years:
        parts.append(f"{years}년")
    if rem_months:
        parts.append(f"{rem_months}개월")
    return " ".join(parts) if parts else "1개월"


def _fv_annuity_due(pmt: float, monthly_rate: float, months: int) -> float:
    """월초 납입(annuity due) 기준 n개월 후 적립액."""
    if months <= 0:
        return 0.0
    if abs(monthly_rate) < 1e-12:
        return pmt * months
    growth = (1 + monthly_rate) ** months
    return pmt * (growth - 1) / monthly_rate * (1 + monthly_rate)


def compute_compound_annual_return(summary: pd.DataFrame) -> Optional[float]:
    """매월 동일 납입 가정 하, 환급금을 만드는 연복리 수익률(%).

    월납입 = 총 납입금 / 납입개월, 월초 납입 적립식(복리)으로
    최종 환급금이 되도록 하는 월이율을 구한 뒤 연환산한다.
    """
    if summary.empty:
        return None

    latest = summary.iloc[-1]
    total_payment = float(latest["cum_payment"])
    refund = float(latest["current_refund"])
    months = payment_duration_months(summary)
    if total_payment <= 0 or refund <= 0:
        return None

    pmt = total_payment / months

    def surplus(monthly_rate: float) -> float:
        return _fv_annuity_due(pmt, monthly_rate, months) - refund

    if surplus(0.0) >= 0:
        return 0.0

    lo, hi = 0.0, 0.05
    if surplus(hi) < 0:
        hi = 0.2
        if surplus(hi) < 0:
            return None

    for _ in range(100):
        mid = (lo + hi) / 2
        if surplus(mid) > 0:
            hi = mid
        else:
            lo = mid

    monthly_rate = (lo + hi) / 2
    return ((1 + monthly_rate) ** 12 - 1) * 100


def _y_range_with_padding(
    values: np.ndarray,
    *,
    pad_ratio: float = 0.2,
    floor_zero: bool = False,
) -> list[float]:
    """데이터 레이블이 잘리지 않도록 y축 범위에 여백을 더한다."""
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return [-1.0, 1.0]
    ymin = float(np.min(clean))
    ymax = float(np.max(clean))
    if floor_zero:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
    span = ymax - ymin
    pad = span * pad_ratio if span > 0 else max(abs(ymax), 1.0) * pad_ratio
    return [ymin - pad, ymax + pad]


def year_start_dates(summary: pd.DataFrame) -> list[pd.Timestamp]:
    """차트 x축 범위에 해당하는 매년 1월 1일 목록."""
    dates = pd.to_datetime(summary["date"])
    if dates.empty:
        return []
    return [
        pd.Timestamp(year, 1, 1)
        for year in range(int(dates.min().year), int(dates.max().year) + 1)
    ]


def render_yearly_chart(yearly: pd.DataFrame, title: str) -> go.Figure:
    """년도별 수익 테이블 데이터를 3단 차트로 시각화한다."""
    is_combined = title == "전체 합산"
    payment_textposition = "bottom center" if is_combined else "top center"
    refund_textposition = "top center" if is_combined else "bottom center"

    year_list = yearly["year"].astype(int).tolist()
    payment = yearly["cum_payment"].astype(float).to_numpy()
    refund = yearly["current_refund"].astype(float).to_numpy()
    rate = yearly["rate"].astype(float).to_numpy()
    profit_yoy = yearly["profit_yoy"].astype(float).to_numpy()
    rate_yoy = yearly["rate_yoy"].astype(float).to_numpy()

    profit_yoy_plot = [None if np.isnan(v) else float(v) for v in profit_yoy]
    rate_yoy_plot = [None if np.isnan(v) else float(v) for v in rate_yoy]
    bar_labels = [
        format_compact_amount(v) if v is not None else ""
        for v in profit_yoy_plot
    ]
    rate_yoy_labels = [
        f"{v:+.2f}%p" if v is not None else ""
        for v in rate_yoy_plot
    ]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.11,
        row_heights=[0.42, 0.28, 0.3],
        subplot_titles=(
            f"{title} · 12/31 기준 금액",
            "수익률 (%)",
            "전년 대비 증가",
        ),
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    fig.add_trace(
        go.Scatter(
            x=year_list,
            y=payment,
            mode="lines+markers+text",
            name="누적 납입금",
            legendgroup="amounts",
            line=dict(color="#4C78A8", width=2),
            text=[format_compact_amount(v) for v in payment],
            textposition=payment_textposition,
            textfont=dict(size=10),
            cliponaxis=False,
            hovertemplate="%{x}년<br>누적 납입금: %{y:,.0f} 원<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=year_list,
            y=refund,
            mode="lines+markers+text",
            name="누적 환급금",
            legendgroup="amounts",
            line=dict(color="#F58518", width=2),
            text=[format_compact_amount(v) for v in refund],
            textposition=refund_textposition,
            textfont=dict(size=10),
            cliponaxis=False,
            hovertemplate="%{x}년<br>누적 환급금: %{y:,.0f} 원<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=year_list,
            y=rate,
            mode="lines+markers+text",
            name="수익률",
            legendgroup="rate",
            line=dict(color="#B279A2", width=2),
            text=[f"{v:.2f}%" for v in rate],
            textposition="top center",
            textfont=dict(size=10),
            cliponaxis=False,
            hovertemplate="%{x}년<br>수익률: %{y:.2f} %<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.add_trace(
        go.Bar(
            x=year_list,
            y=profit_yoy_plot,
            name="전년 대비 수익 증가",
            legendgroup="yoy",
            marker_color="rgba(76, 120, 168, 0.7)",
            text=bar_labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}년<br>수익 증가: %{y:+,.0f} 원<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=year_list,
            y=rate_yoy_plot,
            mode="lines+markers+text",
            name="전년 대비 수익률 증가",
            legendgroup="yoy",
            line=dict(color="#F58518", width=2),
            marker=dict(size=7, color="#F58518"),
            text=rate_yoy_labels,
            textposition="bottom center",
            textfont=dict(size=10, color="#F58518"),
            connectgaps=False,
            cliponaxis=False,
            hovertemplate="%{x}년<br>수익률 증가: %{y:+.2f}%p<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    amount_range = _y_range_with_padding(
        np.concatenate([payment, refund]), pad_ratio=0.22, floor_zero=True
    )
    rate_range = _y_range_with_padding(rate, pad_ratio=0.22, floor_zero=True)
    profit_yoy_clean = profit_yoy[~np.isnan(profit_yoy)]
    rate_yoy_clean = rate_yoy[~np.isnan(rate_yoy)]
    profit_yoy_range = _y_range_with_padding(
        profit_yoy_clean if profit_yoy_clean.size else np.array([0.0]),
        pad_ratio=0.3,
        floor_zero=True,
    )
    rate_yoy_range = _y_range_with_padding(
        rate_yoy_clean if rate_yoy_clean.size else np.array([0.0]),
        pad_ratio=0.35,
        floor_zero=True,
    )

    fig.update_yaxes(
        title_text="금액 (원)",
        range=amount_range,
        automargin=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="수익률 (%)",
        range=rate_range,
        automargin=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="수익 증가 (원)",
        range=profit_yoy_range,
        showgrid=True,
        automargin=True,
        row=3,
        col=1,
    )
    fig.update_yaxes(
        title_text="수익률 증가 (%p)",
        range=rate_yoy_range,
        showgrid=False,
        zeroline=False,
        showline=True,
        automargin=True,
        secondary_y=True,
        row=3,
        col=1,
    )

    xaxis_style = dict(
        title_text="연도",
        tickmode="linear",
        dtick=1,
        tickangle=-45,
        showticklabels=True,
    )
    fig.update_xaxes(**xaxis_style, row=1, col=1)
    fig.update_xaxes(**xaxis_style, row=2, col=1)
    fig.update_xaxes(**xaxis_style, row=3, col=1)

    fig.update_layout(
        hovermode="closest",
        legend=dict(orientation="h", y=-0.16),
        height=980,
        margin=dict(l=20, r=20, t=80, b=60),
        barmode="overlay",
    )
    return fig


def render_chart(
    summary: pd.DataFrame, title: str, period_label: str = "MAX(전체기간)"
) -> go.Figure:
    """수익/수익률 시각화를 위한 2단 그래프를 생성한다.

    상단(금액 그래프): 0부터 위로 원금(파랑), 수익(초록), 손실(빨강) 영역 음영 + 라인.
        - 원금 영역: 0 ~ min(누적 납입금, 누적 환급금) — 회수 가능한 원금
        - 수익 영역: 누적 납입금 ~ 누적 환급금 (환급금이 클 때)
        - 손실 영역: 누적 환급금 ~ 누적 납입금 (납입금이 클 때, 회수 못한 원금)

    하단(수익률 그래프): 0 기준 면적 그래프.

    금액·수익률 그래프는 x축을 분리하고, 날짜는 YYYY-MM-DD 숫자 형식으로 표시한다.
    """
    dates = summary["date"]
    payment = summary["cum_payment"].astype(float).to_numpy()
    refund = summary["current_refund"].astype(float).to_numpy()
    profit = summary["profit"].astype(float).to_numpy()
    rate = summary["rate"].astype(float).to_numpy()
    show_labels = period_label in CHART_PERIODS_WITH_LABELS
    line_mode = "lines+markers+text" if show_labels else "lines+markers"
    payment_text = [format_compact_amount(v) for v in payment] if show_labels else None
    refund_text = [format_compact_amount(v) for v in refund] if show_labels else None
    rate_text = [f"{v:.2f}%" for v in rate] if show_labels else None
    date_hover = f"|{CHART_DATE_FORMAT}"

    principal_top = np.minimum(payment, refund)
    upper_top = np.maximum(payment, refund)
    has_principal = bool(np.any(principal_top > 0))
    has_profit = bool(np.any(refund > payment))
    has_loss = bool(np.any(refund < payment))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, "수익률 (%)"),
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=principal_top,
            mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            fill="tozeroy",
            fillcolor="rgba(76, 120, 168, 0.25)",
            name="원금 영역",
            showlegend=has_principal,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=payment,
            mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=upper_top,
            mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(84, 162, 75, 0.35)",
            name="수익 영역",
            showlegend=has_profit,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=principal_top,
            mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=payment,
            mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            fill="tonexty",
            fillcolor="rgba(220, 60, 60, 0.35)",
            name="손실 영역",
            showlegend=has_loss,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=payment,
            mode=line_mode,
            name="누적 납입금",
            line=dict(color="#4C78A8", width=2),
            text=payment_text,
            textposition="top center",
            textfont=dict(size=10),
            hovertemplate=f"%{{x{date_hover}}}<br>누적 납입금: %{{y:,.0f}} 원<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=refund,
            mode=line_mode,
            name="누적 환급금",
            line=dict(color="#F58518", width=2),
            text=refund_text,
            textposition="bottom center",
            textfont=dict(size=10),
            hovertemplate=f"%{{x{date_hover}}}<br>누적 환급금: %{{y:,.0f}} 원<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=profit,
            mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            name="수익",
            showlegend=False,
            hovertemplate=f"%{{x{date_hover}}}<br>수익: %{{y:+,.0f}} 원<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rate,
            mode=line_mode,
            name="수익률 (%)",
            line=dict(color="#B279A2", width=2),
            text=rate_text,
            textposition="top center",
            textfont=dict(size=10),
            hovertemplate=f"%{{x{date_hover}}}<br>수익률: %{{y:+.2f}} %<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    for year_start in year_start_dates(summary):
        line_style = dict(
            line_dash="dot",
            line_color="rgba(150, 150, 150, 0.55)",
            line_width=1,
            layer="below",
        )
        fig.add_vline(x=year_start, row=1, col=1, **line_style)
        fig.add_vline(x=year_start, row=2, col=1, **line_style)

    fig.update_yaxes(title_text="금액 (원)", row=1, col=1)

    if rate.size > 0:
        rate_min = float(min(np.nanmin(rate), 0.0))
        rate_max = float(max(np.nanmax(rate), 0.0))
        span = rate_max - rate_min
        pad = span * 0.15 if span > 0 else 5.0
        fig.update_yaxes(
            title_text="수익률 (%)",
            range=[rate_min - pad, rate_max + pad],
            row=2,
            col=1,
        )
    else:
        fig.update_yaxes(title_text="수익률 (%)", row=2, col=1)

    xaxis_date_style = dict(
        title_text="날짜",
        tickformat=CHART_DATE_FORMAT,
        hoverformat=CHART_DATE_FORMAT,
        tickangle=-45,
    )
    fig.update_xaxes(**xaxis_date_style, row=1, col=1)
    fig.update_xaxes(**xaxis_date_style, row=2, col=1)

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.22),
        height=860,
        margin=dict(l=20, r=20, t=70, b=40),
    )
    return fig


def format_won(value: float) -> str:
    """금액을 원 단위 문자열로 포맷한다."""
    return f"{int(round(value)):,} 원"


def _authenticate_user_from_config() -> Tuple[bool, Optional[str], Optional[Any]]:
    """config.yaml 기반 로그인 처리 (main_backup.py와 동일한 흐름)."""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = yaml.load(file, Loader=SafeLoader)
    except FileNotFoundError:
        st.error(
            f"⚠️ 'config.yaml' 파일을 찾을 수 없습니다. "
            f"다음 경로에 두세요: `{CONFIG_PATH}`"
        )
        return (False, None, None)

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )
    if not st.session_state.get("_auth_bootstrapped"):
        st.session_state["authentication_status"] = None
        st.session_state["username"] = None
        st.session_state["name"] = None
        st.session_state["email"] = None
        st.session_state["logout"] = True
        st.session_state["_auth_bootstrapped"] = True

    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = None
    if "logout" not in st.session_state:
        st.session_state["logout"] = True

    try:
        authenticator.login(location="unrendered")
    except TypeError:
        pass

    auth_status = st.session_state.get("authentication_status")
    user_name = st.session_state.get("name")
    if not st.session_state.get("authentication_status"):
        try:
            authenticator.login(location="main")
        except TypeError:
            authenticator.login()
        auth_status = st.session_state.get("authentication_status")
        user_name = st.session_state.get("name")
        if auth_status:
            st.session_state["logout"] = None

    return (bool(auth_status), user_name, authenticator)


def main() -> None:
    st.set_page_config(page_title="자산 수익률 추적기", page_icon="💰", layout="wide")

    auth_ok, auth_user_name, authenticator = _authenticate_user_from_config()
    if not auth_ok:
        st.stop()

    with st.sidebar:
        st.markdown("### 자산 수익률 추적기")
        st.markdown(f"**👋 환영합니다, {auth_user_name or '사용자'}님**")
        if authenticator and st.button("로그아웃", key="sidebar_logout_btn"):
            try:
                authenticator.logout(location="unrendered")
            except Exception:
                st.session_state["authentication_status"] = None
                st.session_state["username"] = None
                st.session_state["name"] = None
                st.session_state["email"] = None
            st.session_state["logout"] = True
            st.rerun()

    tab_input, tab_view, tab_chart, tab_yearly = st.tabs(
        ["📝 데이터 입력", "📋 데이터 보기/수정", "📈 수익 그래프", "📅 년도별 수익"]
    )

    with tab_input:
        st.subheader("새 기록 추가")
        df = load_data()
        latest_payments = get_latest_payment_by_asset(df)

        if st.session_state.pop("_reset_amount_fields", False):
            st.session_state["kdb_payment_input"] = format_default_amount(
                latest_payments.get("KDB생명", 0)
            )
            st.session_state["kyobo_payment_input"] = format_default_amount(
                latest_payments.get("교보생명", 0)
            )
            st.session_state["kdb_refund_input"] = ""
            st.session_state["kyobo_refund_input"] = ""
            st.session_state["record_date_input"] = korea_today()
        elif not st.session_state.get("_input_defaults_initialized"):
            st.session_state["kdb_payment_input"] = format_default_amount(
                latest_payments.get("KDB생명", 0)
            )
            st.session_state["kyobo_payment_input"] = format_default_amount(
                latest_payments.get("교보생명", 0)
            )
            st.session_state["kdb_refund_input"] = ""
            st.session_state["kyobo_refund_input"] = ""
            st.session_state["record_date_input"] = korea_today()
            st.session_state["_input_defaults_initialized"] = True
        else:
            for key in AMOUNT_FIELD_KEYS:
                st.session_state.setdefault(key, "")
            st.session_state.setdefault("record_date_input", korea_today())

        st.date_input(
            "날짜",
            min_value=EDITABLE_DATE_MIN,
            max_value=EDITABLE_DATE_MAX,
            key="record_date_input",
        )

        st.markdown("##### KDB생명")
        kdb_col1, kdb_col2 = st.columns(2)
        with kdb_col1:
            st.text_input(
                "KDB생명 누적 납입금 (원)",
                key="kdb_payment_input",
                placeholder="예: 1,200,000",
                on_change=format_amount_field,
                args=("kdb_payment_input",),
            )
        with kdb_col2:
            st.text_input(
                "KDB생명 누적 환급금 (원)",
                key="kdb_refund_input",
                placeholder="예: 1,150,000",
                on_change=format_amount_field,
                args=("kdb_refund_input",),
            )

        st.markdown("##### 교보생명")
        kyobo_col1, kyobo_col2 = st.columns(2)
        with kyobo_col1:
            st.text_input(
                "교보생명 누적 납입금 (원)",
                key="kyobo_payment_input",
                placeholder="예: 2,400,000",
                on_change=format_amount_field,
                args=("kyobo_payment_input",),
            )
        with kyobo_col2:
            st.text_input(
                "교보생명 누적 환급금 (원)",
                key="kyobo_refund_input",
                placeholder="예: 2,250,000",
                on_change=format_amount_field,
                args=("kyobo_refund_input",),
            )

        st.info(
            "💡 동일한 날짜·자산이 이미 존재하면 입력값으로 덮어씁니다.\n\n"
            "- **누적 납입금**: 해당 날짜까지 총 납입한 금액 (그날까지의 총합)\n"
            "- **누적 환급금**: 해당 시점에 환급받을 수 있는 총 금액 (해약환급금)\n"
            "- 한 자산의 납입금/환급금이 모두 0 또는 빈 칸이면 그 자산은 기록되지 않습니다.\n"
            "- 숫자 입력 후 포커스를 옮기면 자동으로 천 단위 콤마(#,##0)로 표시됩니다."
        )

        if st.button("기록 추가", type="primary", key="add_record_btn"):
            record_date = st.session_state["record_date_input"]
            kdb_payment = parse_amount(st.session_state.get("kdb_payment_input"))
            kdb_refund = parse_amount(st.session_state.get("kdb_refund_input"))
            kyobo_payment = parse_amount(st.session_state.get("kyobo_payment_input"))
            kyobo_refund = parse_amount(st.session_state.get("kyobo_refund_input"))

            added_msgs: list[str] = []
            if kdb_payment > 0 or kdb_refund > 0:
                add_record(record_date, "KDB생명", float(kdb_payment), float(kdb_refund))
                added_msgs.append(
                    f"KDB생명 (납입금 {format_won(kdb_payment)}, "
                    f"환급금 {format_won(kdb_refund)})"
                )
            if kyobo_payment > 0 or kyobo_refund > 0:
                add_record(
                    record_date, "교보생명", float(kyobo_payment), float(kyobo_refund)
                )
                added_msgs.append(
                    f"교보생명 (납입금 {format_won(kyobo_payment)}, "
                    f"환급금 {format_won(kyobo_refund)})"
                )

            if added_msgs:
                st.session_state["_last_save_msg"] = (
                    f"{record_date} 기록을 저장했습니다.\n\n- "
                    + "\n- ".join(added_msgs)
                )
                st.session_state["_reset_amount_fields"] = True
                st.rerun()
            else:
                st.warning(
                    "입력한 금액이 모두 0 또는 빈 칸이라 기록을 추가하지 않았습니다."
                )

        if "_last_save_msg" in st.session_state:
            st.success(st.session_state.pop("_last_save_msg"))

    with tab_view:
        st.subheader("기록 보기 및 수정")
        df = load_data()
        if df.empty:
            st.info("아직 입력된 기록이 없습니다. '데이터 입력' 탭에서 추가해주세요.")
        else:
            view_df = df.copy()
            view_df["date"] = view_df["date"].dt.date
            edited = st.data_editor(
                view_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "date": st.column_config.DateColumn(
                        "날짜",
                        min_value=EDITABLE_DATE_MIN,
                        max_value=EDITABLE_DATE_MAX,
                    ),
                    "asset": st.column_config.SelectboxColumn(
                        "자산", options=ASSETS, required=True
                    ),
                    "payment": st.column_config.NumberColumn(
                        "납입금", min_value=0, step=10000, format="localized"
                    ),
                    "refund": st.column_config.NumberColumn(
                        "환급금", min_value=0, step=10000, format="localized"
                    ),
                },
                key="data_editor",
            )
            col_save, col_reset = st.columns([1, 5])
            with col_save:
                if st.button("변경사항 저장", type="primary"):
                    cleaned = edited.dropna(subset=["date", "asset"]).copy()
                    cleaned["date"] = pd.to_datetime(cleaned["date"])
                    cleaned["payment"] = pd.to_numeric(
                        cleaned["payment"], errors="coerce"
                    ).fillna(0)
                    cleaned["refund"] = pd.to_numeric(
                        cleaned["refund"], errors="coerce"
                    ).fillna(0)
                    cleaned = cleaned.sort_values(["asset", "date"]).reset_index(
                        drop=True
                    )
                    save_data(cleaned)
                    st.success("저장했습니다. 새로고침하면 반영됩니다.")
            with col_reset:
                if st.button("모든 기록 삭제"):
                    _delete_all_collection_docs()
                    st.warning("모든 기록을 삭제했습니다.")

    with tab_chart:
        st.subheader("수익 그래프")
        df = load_data()
        if df.empty:
            st.info("그래프를 그리려면 먼저 데이터를 입력해주세요.")
            return

        view_mode = st.radio(
            "보기 모드",
            options=CHART_VIEW_MODES,
            horizontal=True,
        )
        period_label = st.radio(
            "그래프 기간",
            options=CHART_PERIODS,
            index=len(CHART_PERIODS) - 1,
            horizontal=True,
        )
        if view_mode == "합산":
            selected = ASSETS
            chart_title = "전체 합산 수익"
        else:
            selected = [view_mode]
            chart_title = f"{view_mode} 수익"

        summary = build_summary(df, selected)
        if summary.empty:
            st.info(f"{view_mode} 데이터가 없습니다.")
            return

        full_summary = summary.copy()
        profit_changes = compute_profit_changes(full_summary)
        latest_cum_payment = float(full_summary.iloc[-1]["cum_payment"])

        summary = filter_summary_by_period(summary, period_label)
        if summary.empty:
            st.info(f"선택한 기간({period_label})에 표시할 데이터가 없습니다.")
            return

        if period_label != "MAX(전체기간)":
            chart_title = f"{chart_title} · {period_label}"

        st.markdown("##### 수익 증감 (최신 기록 기준)")
        change_cols = st.columns(4)
        for col, (label, change) in zip(change_cols, profit_changes.items()):
            if change is None:
                col.metric(label, "—", help="비교할 기준 데이터가 없습니다.")
            else:
                col.metric(label, format_profit_change(change, latest_cum_payment))

        latest = full_summary.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("누적 납입금", format_won(latest["cum_payment"]))
        col2.metric("누적 환급금", format_won(latest["current_refund"]))
        col3.metric("수익", format_won(latest["profit"]))
        payment_duration = format_payment_duration(full_summary)
        compound_rate = compute_compound_annual_return(full_summary)
        with col4:
            st.metric("수익률", f"{latest['rate']:.2f} %")
            if compound_rate is None:
                st.caption(f"납입기간 {payment_duration} · 복리(연) —")
            else:
                st.caption(
                    f"납입기간 {payment_duration} · 복리(연) {compound_rate:.2f}%"
                )

        st.plotly_chart(
            render_chart(summary, chart_title, period_label),
            use_container_width=True,
        )

        with st.expander("계산 데이터 보기"):
            display_df = summary.copy()
            display_df["date"] = pd.to_datetime(display_df["date"]).dt.date
            display_df.columns = [
                "날짜",
                "누적 납입금",
                "누적 환급금",
                "수익",
                "수익률(%)",
            ]
            st.dataframe(display_df, use_container_width=True)

    with tab_yearly:
        st.subheader("년도별 수익")
        df = load_data()
        if df.empty:
            st.info("데이터를 먼저 입력해주세요.")
        else:
            yearly_view_mode = st.radio(
                "보기 모드",
                options=CHART_VIEW_MODES,
                horizontal=True,
                key="yearly_view_mode",
            )
            if yearly_view_mode == "합산":
                yearly_selected = ASSETS
            else:
                yearly_selected = [yearly_view_mode]

            yearly_summary = build_summary(df, yearly_selected)
            if yearly_summary.empty:
                st.info(f"{yearly_view_mode} 데이터가 없습니다.")
            else:
                yearly_table = build_yearly_summary(yearly_summary)
                if yearly_view_mode == "합산":
                    yearly_chart_title = "전체 합산"
                else:
                    yearly_chart_title = yearly_view_mode

                st.plotly_chart(
                    render_yearly_chart(yearly_table, yearly_chart_title),
                    use_container_width=True,
                )

                display_yearly = yearly_table.copy()
                display_yearly["cum_payment"] = display_yearly["cum_payment"].map(
                    format_won
                )
                display_yearly["current_refund"] = display_yearly[
                    "current_refund"
                ].map(format_won)
                display_yearly["profit"] = display_yearly["profit"].map(format_won)
                display_yearly["rate"] = display_yearly["rate"].map(
                    lambda v: f"{v:.2f} %"
                )
                display_yearly["profit_yoy"] = display_yearly["profit_yoy"].map(
                    lambda v: format_signed_won(v) if pd.notna(v) else "—"
                )
                display_yearly["rate_yoy"] = display_yearly["rate_yoy"].map(
                    lambda v: f"{v:+.2f}%p" if pd.notna(v) else "—"
                )
                display_yearly.columns = [
                    "연도",
                    "누적 납입금 (12/31)",
                    "누적 환급금 (12/31)",
                    "수익 (12/31)",
                    "수익률",
                    "전년 대비 수익 증가",
                    "전년 대비 수익률 증가",
                ]
                st.caption("각 연도 12월 31일 기준 보간값입니다.")
                st.dataframe(display_yearly, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

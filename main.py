"""자산 수익률 추적 웹 앱.

KDB생명, 교보생명의 납입금/환급금을 날짜별로 기록하고
수익을 꺾은선 그래프로 시각화한다.

실행 방법:
    streamlit run main.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Optional, Tuple

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
                "date": d.get("date"),
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
        df["date"] = pd.to_datetime(df["date"])
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


def render_chart(summary: pd.DataFrame, title: str) -> go.Figure:
    """수익/수익률 시각화를 위한 2단 그래프를 생성한다.

    상단(금액 그래프): 0부터 위로 원금(파랑), 수익(초록), 손실(빨강) 영역 음영 + 라인.
        - 원금 영역: 0 ~ min(누적 납입금, 누적 환급금) — 회수 가능한 원금
        - 수익 영역: 누적 납입금 ~ 누적 환급금 (환급금이 클 때)
        - 손실 영역: 누적 환급금 ~ 누적 납입금 (납입금이 클 때, 회수 못한 원금)

    하단(수익률 그래프): 0 기준 면적 그래프.

    shared_xaxes=True로 두 그래프의 x축 범위와 plot area의 가로 폭이
    동일하게 정렬되어 같은 시점을 위/아래로 바로 비교할 수 있다.
    """
    dates = summary["date"]
    payment = summary["cum_payment"].astype(float).to_numpy()
    refund = summary["current_refund"].astype(float).to_numpy()
    profit = summary["profit"].astype(float).to_numpy()
    rate = summary["rate"].astype(float).to_numpy()

    principal_top = np.minimum(payment, refund)
    upper_top = np.maximum(payment, refund)
    has_principal = bool(np.any(principal_top > 0))
    has_profit = bool(np.any(refund > payment))
    has_loss = bool(np.any(refund < payment))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.18,
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
            mode="lines+markers",
            name="누적 납입금",
            line=dict(color="#4C78A8", width=2),
            hovertemplate="누적 납입금: %{y:,.0f} 원<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=refund,
            mode="lines+markers",
            name="누적 환급금",
            line=dict(color="#F58518", width=2),
            hovertemplate="누적 환급금: %{y:,.0f} 원<extra></extra>",
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
            hovertemplate="수익: %{y:+,.0f} 원<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=rate,
            mode="lines+markers",
            name="수익률 (%)",
            line=dict(color="#B279A2", width=2),
            hovertemplate="수익률: %{y:+.2f} %<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

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

    fig.update_xaxes(title_text="날짜", row=2, col=1)

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
        height=820,
        margin=dict(l=20, r=20, t=70, b=20),
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

    tab_input, tab_view, tab_chart = st.tabs(
        ["📝 데이터 입력", "📋 데이터 보기/수정", "📈 수익 그래프"]
    )

    with tab_input:
        st.subheader("새 기록 추가")
        if st.session_state.pop("_reset_amount_fields", False):
            for key in AMOUNT_FIELD_KEYS:
                st.session_state[key] = ""
        else:
            for key in AMOUNT_FIELD_KEYS:
                st.session_state.setdefault(key, "")

        st.date_input(
            "날짜",
            value=date.today(),
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
            options=["KDB생명", "교보생명", "합산"],
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

        latest = summary.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("누적 납입금", format_won(latest["cum_payment"]))
        col2.metric("누적 환급금", format_won(latest["current_refund"]))
        col3.metric(
            "수익",
            format_won(latest["profit"]),
            delta=f"{latest['profit']:+,.0f} 원",
        )
        col4.metric("수익률", f"{latest['rate']:.2f} %")

        st.plotly_chart(
            render_chart(summary, chart_title), use_container_width=True
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


if __name__ == "__main__":
    main()

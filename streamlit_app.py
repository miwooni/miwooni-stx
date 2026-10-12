# -*- coding: utf-8 -*-
"""
==============================================================================
 MIWOONI Trading Dashboard · Hardened Build v2.0
 업비트 KRW 마켓 파동 매매 + 리스크 관리 대시보드 (Streamlit Community Cloud)

 설계 원칙
  1) 임계값은 전부 상수로 선언하고 근거를 주석으로 명시 (매직넘버 제거)
  2) 지표는 Wilder 원전 / 업계 표준 정의 사용 (RSI·ATR·ADX는 RMA 스무딩)
  3) 신호는 ADX로 추세장/횡보장을 먼저 판별한 뒤 가중치를 바꾸는 국면 인식 구조
  4) 리스크(ATR 손절·목표, 포지션 사이징, 집중도, MDD, 변동성)를 신호와 동급으로 취급
  5) 자격증명은 코드에 두지 않고 st.secrets / 환경변수에서만 읽음

 ⚠️ 본 대시보드는 정보 제공 및 의사결정 보조용이며 투자 자문이 아닙니다.
    모든 매매 판단과 그 결과에 대한 책임은 사용자 본인에게 있습니다.
==============================================================================
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

try:  # 자동 새로고침(선택 의존성) — 없어도 앱이 죽지 않도록 방어
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False


# =============================================================================
# 1. 설정 상수  —  모든 임계값은 여기서 한 번만 정의한다
# =============================================================================
BASE_MARKET = "KRW-SAND"                     # 기축(비교 기준) 자산
MARKETS = ["KRW-SAND", "KRW-ENA", "KRW-ONDO", "KRW-STX", "KRW-HBAR", "KRW-DOGE"]

# 보유 수량(개). 0.0 인 종목은 "미보유 / 관찰 대상"
HOLDINGS = {
    "KRW-SAND": 67645.82069552,
    "KRW-ENA": 20000.0,
    "KRW-ONDO": 15000.0,
    "KRW-STX": 15604.4864,
    "KRW-HBAR": 62216.0,
    "KRW-DOGE": 61194.37067502,
}

# 수량 목표(개). 0 = 목표 미설정. 사이드바에서 수정 가능
DEFAULT_GOALS = {
    "KRW-SAND": 100_000.0,
    "KRW-ENA": 0.0,
    "KRW-ONDO": 0.0,
    "KRW-STX": 0.0,
    "KRW-HBAR": 100_000.0,
    "KRW-DOGE": 0.0,
}

TOTAL_INVESTMENT = 65_000_000        # 총 매수 원금(원) — SAND 기준 평가손익 산출
UPBIT_FEE = 0.0005                   # 업비트 KRW 마켓 기본 수수료 0.05% (매수/매도 각 1회)
SESSION_TIMEOUT_MIN = 120            # 로그인 세션 만료(분)
LOGIN_MAX_FAILS = 5                  # 로그인 실패 허용 횟수(세션 기준)
LOGIN_FAIL_SLEEP = 1.0               # 실패 1회당 지연(초) — 무차별 대입 완화

TIMEFRAMES = {1: "1분", 3: "3분", 5: "5분", 15: "15분", 60: "60분", 240: "240분"}
# 상위 시간프레임(멀티 타임프레임 추세 확인용): 자기보다 3~4배 긴 주기
HTF_MAP = {1: 5, 3: 15, 5: 15, 15: 60, 60: 240, 240: 240}
REFRESH_MS = 10_000                  # 자동 새로고침 주기(ms)

# --- 네트워크 / 캐시 ---
UPBIT_BASE = "https://api.upbit.com"
HTTP_TIMEOUT = 8                     # 요청 타임아웃(초)
MAX_RETRIES = 2                      # 재시도 횟수(총 3회 시도)
RATE_LIMIT_PER_SEC = 8.0             # 업비트 공식 제한(초당 10회)의 80% → 429 회피
CACHE_TTL_PRICE = 10                 # 시세 캐시(초)
CACHE_TTL_CANDLE = 30                # 캔들 캐시(초)
OHLCV_COUNT = 200                    # 업비트 분봉 API 1회 최대 200개 (초과 요청 시 오류)
API_FAIL_COOLDOWN = 60               # API 실패 후 재시도 억제 시간(초) — 지역차단 시 폭주 방지

# --- 지표 파라미터 (Wilder / 업계 표준) ---
HMA_PERIOD = 16                      # Hull MA 기본 16 (HMA 원전 권장값)
HMA3_PERIOD = 3                      # 단기 크로스용
RSI_PERIOD = 14                      # Wilder 원전 기본 14
RSI_OB = 70                          # 과매수(70 이상) — Wilder 기준
RSI_OS = 30                          # 과매도(30 이하)
RSI_TREND_PIVOT = 50                 # 추세장에서 강세/약세 분기선
STOCH_RSI_PERIOD = 14                # Stochastic RSI 계산에 쓰는 RSI 구간
STOCH_PERIOD = 14                    # RSI의 최고/최저 구간
STOCH_K_SMOOTH = 3                   # %K 스무딩
STOCH_D_SMOOTH = 3                   # %D 스무딩
STOCH_OB = 80                        # Stoch RSI 과매수
STOCH_OS = 20                        # Stoch RSI 과매도
ADX_PERIOD = 14                      # Wilder ADX 기본 14
ADX_TREND_MIN = 25                   # 25 이상 = 추세장 (Wilder의 추세 판별 임계)
ADX_STRONG = 40                      # 40 이상 = 강한 추세
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9   # 표준 MACD(12,26,9)
BB_PERIOD, BB_STD = 20, 2.0          # 볼린저 20기간 ±2σ (표준)
BB_SQUEEZE_PCT = 20                  # 볼린저 밴드폭이 최근 100봉 중 하위 20% = 스퀴즈
CCI_PERIOD = 20                      # CCI 표준 20
CCI_OB, CCI_OS = 100, -100           # CCI 과매수/과매도 기준
ATR_PERIOD = 14                      # Wilder ATR 14
ATR_STOP_MULT = 2.0                  # 손절 = 진입가 ∓ 2 ATR (추세추종 표준 2~3)
ATR_TARGET_MULT = 3.0                # 목표 = 진입가 ± 3 ATR → 최소 R:R 1.5
VWAP_WINDOW = 20                     # 롤링 VWAP 구간(24시간 시장이라 세션 VWAP 대신 롤링)
OBV_EMA_SPAN = 10                    # OBV 추세 판정용 EMA
VOL_ZSCORE_LOOKBACK = 20             # 거래량 z-score 기준 구간
VOL_Z_HOT = 1.5                      # z >= 1.5 → 거래량 급증
VOL_Z_EXTREME = 3.0                  # z >= 3.0 → 이상 급증(뉴스성)
RISK_WINDOW = 120                    # 변동성·MDD 산출 구간(봉)
ANN_MINUTES = 365 * 24 * 60          # 연율화 계수(분 단위) — 24시간 시장 가정
RISK_PER_TRADE = 0.01                # 1회 매매 리스크 = 자산의 1% (고정비율 방식)
CONCENTRATION_WARN = 0.40            # 단일 자산 비중 40% 초과 → 집중 경고
CONCENTRATION_HIGH = 0.60            # 60% 초과 → 심각
MIN_RR = 1.5                         # 최소 허용 손익비
MIN_BARS_FOR_SIGNAL = 60             # 신호 산출 최소 봉 수(EMA55 등 확보)

# --- 신호 등급 임계 (복합점수 -100 ~ +100) ---
SCORE_STRONG = 55                    # |점수| >= 55 → 강한 신호
SCORE_MILD = 25                      # |점수| >= 25 → 유효 신호, 미만은 관망

# --- 패턴 감지 ---
PIVOT_LEFT = 2                       # 프랙탈 피벗 좌측 봉 수
PIVOT_RIGHT = 2                      # 우측 봉 수
PIVOT_TOL = 0.008                    # 동일 레벨로 인정하는 허용 오차(0.8%)
MIN_SWING = 0.010                    # 넥라인/스윙 최소 폭(1.0%) — 노이즈 패턴 배제

# --- 알림 쿨다운(분): 같은 알림 반복 전송 방지 ---
ALERT_COOLDOWN = {
    "target": 10, "reentry": 15, "pattern": 30, "rsi_ob": 30, "rsi_os": 30,
    "stoch": 30, "vol_spike": 30, "atr_stop": 30, "grade": 30, "move": 20,
}
TG_DEDUP_WINDOW = 300                # 텔레그램 동일 메시지 재전송 금지 시간(초)
MOVE_ALERT_PCT = 5.0                 # 단기 등락률 ±5% 이상이면 알림


# =============================================================================
# 2. 시크릿 로딩 — 자격증명은 절대 코드에 하드코딩하지 않는다
#    (Streamlit Cloud: Settings → Secrets / 로컬: .streamlit/secrets.toml)
# =============================================================================
def _secret(key: str, default: str | None = None) -> str | None:
    """st.secrets → 환경변수 순서로 읽는다. 시크릿 파일이 없어도 예외로 죽지 않는다."""
    try:
        value = st.secrets[key]          # 키 없으면 여기서 예외
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    except Exception:
        pass
    env = os.environ.get(key)
    return env.strip() if env else default


def parse_number(x, default: float = 0.0) -> float:
    """문자열/숫자를 안전하게 float 로 변환.

    원본 코드의 버그: 부호(-)를 제거해 -12.5 를 12.5 로 만들었다.
    여기서는 천단위 쉼표·공백·통화기호만 제거하고 부호는 그대로 보존한다.
    """
    if x is None:
        return default
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        cleaned = x.replace(",", "").replace(" ", "").replace("원", "").replace("₩", "")
        try:
            return float(cleaned)
        except ValueError:
            return default
    return default


APP_ID = _secret("APP_ID", "miwooni")
APP_PW = _secret("APP_PW")
TELEGRAM_BOT_TOKEN = _secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _secret("TELEGRAM_CHAT_ID")


# =============================================================================
# 3. 업비트 데이터 계층 (요청 제한 준수 + 재시도 + 명시적 예외)
# =============================================================================
class UpbitError(RuntimeError):
    """업비트 API 호출 실패(네트워크 / HTTP / 지역 차단 / 파싱)."""


class RateLimiter:
    """초당 호출 수를 보장하는 최소 간격 제한기(프로세스 전역)."""

    def __init__(self, per_second: float):
        self.min_interval = 1.0 / max(per_second, 0.1)
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self.min_interval:
                time.sleep(self.min_interval - delta)
            self._last = time.monotonic()


_LIMITER = RateLimiter(RATE_LIMIT_PER_SEC)


def _upbit_get(path: str, params: dict, timeout: int = HTTP_TIMEOUT):
    """업비트 REST GET. 재시도/요청제한/타임아웃을 모두 처리하고 실패 시 UpbitError."""
    last_error = "알 수 없는 오류"
    for attempt in range(MAX_RETRIES + 1):
        _LIMITER.wait()
        try:
            resp = requests.get(
                f"{UPBIT_BASE}{path}", params=params, timeout=timeout,
                headers={"Accept": "application/json"},
            )
        except requests.RequestException as exc:
            last_error = f"네트워크 오류: {exc.__class__.__name__}"
            time.sleep(0.4 * (attempt + 1))
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                last_error = "응답 JSON 파싱 실패"
                continue
        if resp.status_code in (418, 429):          # 업비트 요청 제한
            last_error = f"요청 제한(HTTP {resp.status_code})"
            time.sleep(1.0 * (attempt + 1))
            continue
        # 4xx/5xx 는 재시도해도 결과가 같을 가능성이 높으므로 즉시 중단(5xx만 1회 재시도)
        last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
        if resp.status_code >= 500 and attempt < MAX_RETRIES:
            time.sleep(0.5 * (attempt + 1))
            continue
        break
    raise UpbitError(last_error)


def _mark_api_ok() -> None:
    st.session_state["api_error"] = None
    st.session_state["api_fail_ts"] = None


def _mark_api_fail(message: str) -> None:
    st.session_state["api_error"] = message
    st.session_state["api_fail_ts"] = datetime.now()


def api_blocked() -> bool:
    """최근 실패가 쿨다운 이내면 네트워크 호출을 건너뛴다(지역 차단 시 폭주 방지)."""
    ts = st.session_state.get("api_fail_ts")
    return bool(ts and (datetime.now() - ts) < timedelta(seconds=API_FAIL_COOLDOWN))


@st.cache_data(ttl=CACHE_TTL_PRICE, show_spinner=False)
def _ticker_cached(markets_key: str) -> dict:
    """6개 마켓 시세를 '1회 배치 호출'로 가져온다(호출 수 최소화)."""
    rows = _upbit_get("/v1/ticker", {"markets": markets_key}, timeout=HTTP_TIMEOUT)
    return {row["market"]: row for row in rows}


def get_prices() -> tuple[dict, str | None]:
    """(시세딕셔너리, 오류메시지) 반환. 실패해도 앱은 계속 렌더링된다."""
    if api_blocked():
        return {}, st.session_state.get("api_error", "API 일시 차단")
    try:
        data = _ticker_cached(",".join(MARKETS))
        _mark_api_ok()
        return data, None
    except Exception as exc:                       # noqa: BLE001 - 사용자에게 그대로 표시
        _mark_api_fail(str(exc))
        return {}, str(exc)


@st.cache_data(ttl=CACHE_TTL_CANDLE, show_spinner=False)
def _ohlcv_cached(market: str, unit: int, count: int) -> pd.DataFrame:
    """캔들 조회 + 지표 산출까지 수행한 순수 함수(세션 상태를 건드리지 않는다)."""
    count = int(min(max(count, 30), OHLCV_COUNT))
    raw = _upbit_get(f"/v1/candles/minutes/{unit}", {"market": market, "count": count})

    df = pd.DataFrame(raw)
    if df.empty:
        raise UpbitError(f"{market}: 캔들 데이터 없음")

    need = ["candle_date_time_kst", "opening_price", "high_price",
            "low_price", "trade_price", "candle_acc_trade_volume"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise UpbitError(f"{market}: 응답 컬럼 누락 {missing}")

    df = df[need].iloc[::-1].reset_index(drop=True)      # 업비트는 최신순 → 과거순 정렬
    df.columns = ["datetime", "open", "high", "low", "close", "volume"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if len(df) < 30:
        raise UpbitError(f"{market}: 유효 봉 부족({len(df)})")
    return add_indicators(df)


def load_ohlcv(market: str, unit: int, count: int = OHLCV_COUNT) -> pd.DataFrame:
    """세션 상태 기록까지 담당하는 래퍼. 실패 시 빈 DataFrame(앱은 계속 동작)."""
    if api_blocked():
        return pd.DataFrame()
    try:
        df = _ohlcv_cached(market, unit, count)
        _mark_api_ok()
        return df
    except Exception as exc:                       # noqa: BLE001
        _mark_api_fail(str(exc))
        return pd.DataFrame()


@st.cache_data(ttl=CACHE_TTL_CANDLE, show_spinner=False)
def _htf_trend_cached(market: str, htf_unit: int) -> int:
    """상위 시간프레임 추세: +1 상승 / -1 하락 / 0 혼조 (멀티 타임프레임 확인)."""
    try:
        df = _ohlcv_cached(market, htf_unit, OHLCV_COUNT)
    except Exception:                              # noqa: BLE001
        return 0
    if len(df) < MIN_BARS_FOR_SIGNAL:
        return 0
    c = df.iloc[-1]
    up = (c["close"] > c["EMA21"]) and (c["EMA21"] > c["EMA55"]) and (c["HMA3"] > c["HMA"])
    dn = (c["close"] < c["EMA21"]) and (c["EMA21"] < c["EMA55"]) and (c["HMA3"] < c["HMA"])
    return 1 if up else (-1 if dn else 0)


def htf_trend(market: str, unit: int) -> int:
    return _htf_trend_cached(market, HTF_MAP.get(unit, unit))


# =============================================================================
# 4. 기술적 지표 — Wilder 원전 / 업계 표준 정의
# =============================================================================
def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda x: float(np.dot(x, weights) / weights.sum()), raw=True
    )


def hma(series: pd.Series, period: int = HMA_PERIOD) -> pd.Series:
    """Hull Moving Average — 후행성 완화용. sqrt 기간 WMA로 2차 평활."""
    half = max(int(period / 2), 1)
    root = max(int(np.sqrt(period)), 1)
    return wma(2 * wma(series, half) - wma(series, period), root)


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder 스무딩(RMA) = alpha 1/n 지수이동평균. RSI/ATR/ADX의 표준 기반."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def wilder_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder RSI. 원본은 SMA를 써서 값이 왜곡됐다(표준과 최대 수 포인트 차이)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)   # 손실 0 → 과매수 극단
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)     # 상승 0 → 과매도 극단
    out = out.mask((avg_gain == 0) & (avg_loss == 0), 50.0)   # 완전 무변동 → 중립
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift()
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    return rma(true_range(df), period)


def adx_dmi(df: pd.DataFrame, period: int = ADX_PERIOD):
    """Wilder ADX/DMI. ADX >= 25 를 추세장 판별 기준으로 사용한다."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    atr_ = rma(true_range(df), period)
    plus_di = 100.0 * rma(plus_dm, period) / atr_.replace(0, np.nan)
    minus_di = 100.0 * rma(minus_dm, period) / atr_.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = rma(dx.fillna(0.0), period)
    return plus_di, minus_di, adx


def macd(close: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger(close: pd.Series, period: int = BB_PERIOD, std_mult: float = BB_STD):
    ma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    return ma, ma + std_mult * std, ma - std_mult * std


def cci(df: pd.DataFrame, period: int = CCI_PERIOD) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def stoch_rsi(close: pd.Series, rsi_period: int = RSI_PERIOD,
              stoch_period: int = STOCH_PERIOD,
              k_smooth: int = STOCH_K_SMOOTH, d_smooth: int = STOCH_D_SMOOTH):
    """Stochastic RSI: RSI를 다시 0~100으로 정규화 → 횡보장 과열/침체 판독."""
    r = wilder_rsi(close, rsi_period)
    lowest = r.rolling(stoch_period).min()
    highest = r.rolling(stoch_period).max()
    raw = (r - lowest) / (highest - lowest).replace(0, np.nan) * 100.0
    raw = raw.fillna(50.0)                     # 구간 변동 없음 → 중립값
    k_line = raw.rolling(k_smooth).mean()
    return k_line, k_line.rolling(d_smooth).mean()


def rolling_vwap(df: pd.DataFrame, window: int = VWAP_WINDOW) -> pd.Series:
    """롤링 VWAP(거래량 가중 평균가). 암호화폐는 24시간 거래라 세션 VWAP 대신 롤링 사용."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = (tp * df["volume"]).rolling(window).sum()
    vol = df["volume"].rolling(window).sum().replace(0, np.nan)
    return pv / vol


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume: 가격 방향에 거래량을 누적 → 매집/분산 판독."""
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV → 전체 지표 컬럼 부착(원본 df 는 보존)."""
    out = df.copy()
    out["HL2"] = (out["high"] + out["low"]) / 2.0

    out["HMA"] = hma(out["HL2"], HMA_PERIOD)
    out["HMA3"] = hma(out["HL2"], HMA3_PERIOD)
    out["EMA21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["EMA55"] = out["close"].ewm(span=55, adjust=False).mean()
    out["EMA200"] = out["close"].ewm(span=200, adjust=False).mean()

    out["RSI"] = wilder_rsi(out["close"], RSI_PERIOD)
    out["STOCH_K"], out["STOCH_D"] = stoch_rsi(out["close"])
    out["MACD_line"], out["Signal_line"], out["MACD_hist"] = macd(out["close"])

    ma, upper, lower = bollinger(out["close"])
    out["BB_ma"], out["BB_upper"], out["BB_lower"] = ma, upper, lower
    band_range = (upper - lower).replace(0, np.nan)
    out["BB_WIDTH"] = band_range / ma.replace(0, np.nan)          # 밴드폭(스퀴즈 판정)
    out["BB_PCTB"] = (out["close"] - lower) / band_range          # %B: 0=하한, 1=상한
    out["BB_WIDTH_RANK"] = out["BB_WIDTH"].rolling(100).rank(pct=True) * 100.0

    out["CCI"] = cci(out, CCI_PERIOD)
    out["TR"] = true_range(out)
    out["ATR"] = rma(out["TR"], ATR_PERIOD)
    out["ATR_PCT"] = out["ATR"] / out["close"].replace(0, np.nan) * 100.0

    out["PLUS_DI"], out["MINUS_DI"], out["ADX"] = adx_dmi(out, ADX_PERIOD)

    out["VWAP"] = rolling_vwap(out, VWAP_WINDOW)
    out["OBV"] = obv(out)
    out["OBV_EMA"] = out["OBV"].ewm(span=OBV_EMA_SPAN, adjust=False).mean()
    obv_scale = (out["volume"].rolling(VOL_ZSCORE_LOOKBACK).mean() * 5).replace(0, np.nan)
    out["OBV_SLOPE"] = (out["OBV_EMA"] - out["OBV_EMA"].shift(5)) / obv_scale

    out["VOL_MA"] = out["volume"].rolling(VOL_ZSCORE_LOOKBACK).mean()
    out["VOL_STD"] = out["volume"].rolling(VOL_ZSCORE_LOOKBACK).std(ddof=0)
    out["VOL_Z"] = (out["volume"] - out["VOL_MA"]) / out["VOL_STD"].replace(0, np.nan)

    out["RET"] = np.log(out["close"] / out["close"].shift())
    out["DRAWDOWN"] = out["close"] / out["close"].cummax() - 1.0
    return out


# =============================================================================
# 5. 국면 판별 + 가중치 기반 복합 신호 엔진
#    원본은 "가산 점수"라 추세장과 횡보장을 동일 기준으로 채점했다.
#    여기서는 ADX로 국면을 먼저 정하고, 국면별로 가중치 세트를 바꾼다.
#    각 팩터는 -1.0 ~ +1.0 로 정규화되므로 총점은 -100 ~ +100 이다.
# =============================================================================
REGIME_STRONG_TREND = "강한 추세"
REGIME_TREND = "추세"
REGIME_RANGE = "횡보"
REGIME_VOLATILE = "급변동"

# 국면별 팩터 가중치 (합계 100 기준). 추세장은 추세추종, 횡보장은 평균회귀에 무게.
WEIGHTS_TREND = {
    "HMA 크로스": 14, "EMA 배열": 12, "MACD": 11, "ADX/DI": 14,
    "VWAP 위치": 9, "OBV 추세": 9, "RSI 추세선": 7, "볼린저 %B": 6,
    "CCI": 6, "Stoch RSI": 4, "거래량 z": 4, "상위TF 정합": 4,
}
WEIGHTS_RANGE = {
    "HMA 크로스": 6, "EMA 배열": 5, "MACD": 6, "ADX/DI": 4,
    "VWAP 위치": 11, "OBV 추세": 6, "RSI 추세선": 8, "볼린저 %B": 20,
    "CCI": 11, "Stoch RSI": 13, "거래량 z": 6, "상위TF 정합": 4,
}
# 급변동 국면에서는 신호 신뢰도를 낮춘다(가중치를 전부 축소하고 리스크에 위임)
WEIGHTS_VOLATILE = {k: max(int(v * 0.6), 2) for k, v in WEIGHTS_TREND.items()}


def classify_regime(latest: pd.Series, atr_rank: float) -> tuple[str, str]:
    """ADX 와 변동성 순위로 시장 국면을 판정한다. (국면, 근거문자열) 반환."""
    adx = float(latest.get("ADX") or 0.0)
    plus_di = float(latest.get("PLUS_DI") or 0.0)
    minus_di = float(latest.get("MINUS_DI") or 0.0)
    di_dir = "상승 우위" if plus_di > minus_di else "하락 우위"

    if atr_rank >= 90 and adx < ADX_TREND_MIN:
        return REGIME_VOLATILE, f"ADX {adx:.1f} + ATR 상위 {atr_rank:.0f}% → 방향성 없는 급변동"
    if adx >= ADX_STRONG:
        return REGIME_STRONG_TREND, f"ADX {adx:.1f} ≥ {ADX_STRONG} → 강한 추세({di_dir})"
    if adx >= ADX_TREND_MIN:
        return REGIME_TREND, f"ADX {adx:.1f} ≥ {ADX_TREND_MIN} → 추세 진행({di_dir})"
    return REGIME_RANGE, f"ADX {adx:.1f} < {ADX_TREND_MIN} → 횡보(평균회귀 우위)"


def _clip(v, lo: float = -1.0, hi: float = 1.0) -> float:
    """None/NaN → 0(중립)으로 처리한 뒤 [-1, 1] 로 자른다."""
    if v is None:
        return 0.0
    try:
        if pd.isna(v):
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    return float(max(lo, min(hi, float(v))))


def score_factors(df: pd.DataFrame, htf: int, regime: str) -> tuple[list[dict], float, float]:
    """팩터별 (이름, 가중치, -1~1 값, 기여도, 근거) 산출. (factors, 총점, 신뢰도) 반환."""
    if regime in (REGIME_STRONG_TREND, REGIME_TREND):
        weights = WEIGHTS_TREND
    elif regime == REGIME_RANGE:
        weights = WEIGHTS_RANGE
    else:
        weights = WEIGHTS_VOLATILE

    c = df.iloc[-1]
    p = df.iloc[-2]
    close = float(c["close"])
    rsi = float(c["RSI"]) if pd.notna(c["RSI"]) else 50.0
    stoch_k = float(c["STOCH_K"]) if pd.notna(c["STOCH_K"]) else 50.0
    pctb = float(c["BB_PCTB"]) if pd.notna(c["BB_PCTB"]) else 0.5
    cci_v = float(c["CCI"]) if pd.notna(c["CCI"]) else 0.0
    adx = float(c["ADX"]) if pd.notna(c["ADX"]) else 0.0
    plus_di = float(c["PLUS_DI"]) if pd.notna(c["PLUS_DI"]) else 0.0
    minus_di = float(c["MINUS_DI"]) if pd.notna(c["MINUS_DI"]) else 0.0
    vwap = float(c["VWAP"]) if pd.notna(c["VWAP"]) else close
    vol_z = float(c["VOL_Z"]) if pd.notna(c["VOL_Z"]) else 0.0
    atr_pct = float(c["ATR_PCT"]) if pd.notna(c["ATR_PCT"]) else 0.0

    factors: list[dict] = []

    def add(name: str, value: float, reason: str) -> None:
        w = weights.get(name, 0)
        factors.append({
            "팩터": name, "가중치": w, "판독값": round(_clip(value), 3),
            "기여도": round(_clip(value) * w, 2), "근거": reason,
        })

    # 1) HMA 크로스: 단기(3) > 기본(16) 이면 상승 모멘텀
    hma_gap = 0.0
    if pd.notna(c["HMA"]) and float(c["HMA"]) > 0:
        hma_gap = (float(c["HMA3"]) - float(c["HMA"])) / float(c["HMA"])
    add("HMA 크로스", _clip(hma_gap / 0.006), f"HMA3-HMA 괴리 {hma_gap * 100:+.2f}%")

    # 2) EMA 배열: EMA21 vs EMA55 (중기 추세 정렬)
    ema_gap = 0.0
    if pd.notna(c["EMA21"]) and pd.notna(c["EMA55"]) and float(c["EMA55"]) > 0:
        ema_gap = (float(c["EMA21"]) - float(c["EMA55"])) / float(c["EMA55"])
    add("EMA 배열", _clip(ema_gap / 0.010), f"EMA21-EMA55 괴리 {ema_gap * 100:+.2f}%")

    # 3) MACD: 히스토그램 부호 + 확대/축소
    hist, prev_hist = float(c["MACD_hist"]), float(p["MACD_hist"])
    hist_norm = hist / (atr_pct / 100 * close + 1e-9) if close else 0.0
    slope_bonus = 0.25 if hist > prev_hist else -0.25
    add("MACD", _clip(hist_norm * 40 + slope_bonus),
        f"히스토그램 {hist:+,.4f} ({'확대' if hist > prev_hist else '축소'})")

    # 4) ADX/DI: 추세 강도 × 방향
    di_spread = (plus_di - minus_di) / 100.0
    strength = min(adx / ADX_STRONG, 1.0) if adx else 0.0
    add("ADX/DI", _clip(di_spread * strength * 2.0),
        f"ADX {adx:.1f} / +DI {plus_di:.1f} -DI {minus_di:.1f}")

    # 5) VWAP 위치: 기관 평균단가 위/아래
    vwap_dev = (close - vwap) / vwap if vwap else 0.0
    add("VWAP 위치", _clip(vwap_dev / 0.008), f"VWAP 대비 {vwap_dev * 100:+.2f}%")

    # 6) OBV 추세: 매집(상승)/분산(하락)
    obv_slope = float(c["OBV_SLOPE"]) if pd.notna(c["OBV_SLOPE"]) else 0.0
    add("OBV 추세", _clip(obv_slope * 6), f"OBV 기울기 {obv_slope:+.3f}")

    # 7) RSI 추세선: 50 기준 강세/약세 (30/70 극단은 별도 가산)
    rsi_val = _clip((rsi - RSI_TREND_PIVOT) / 20.0)
    if rsi >= RSI_OB:
        rsi_val -= 0.5                      # 과매수는 추세 지속이어도 리스크 가산
    elif rsi <= RSI_OS:
        rsi_val += 0.5
    add("RSI 추세선", rsi_val, f"RSI {rsi:.1f}")

    # 8) 볼린저 %B: 추세장은 상단 돌파=강세, 횡보장은 반전 신호로 해석
    if regime == REGIME_RANGE:
        bb_val = _clip((0.5 - pctb) * 2.2)      # 0.5 중심 역방향(평균회귀)
    else:
        bb_val = _clip((pctb - 0.5) * 1.6)      # 0.5 중심 순방향(추세 지속)
    add("볼린저 %B", bb_val, f"%B {pctb:.2f} (0=하한, 1=상한)")

    # 9) CCI
    add("CCI", _clip(cci_v / CCI_OB), f"CCI {cci_v:+.1f}")

    # 10) Stoch RSI: 단기 과열/침체 — 횡보장에서 비중이 커진다
    if stoch_k >= STOCH_OB:
        st_val = -_clip((stoch_k - STOCH_OB) / (100 - STOCH_OB) + 0.6)
    elif stoch_k <= STOCH_OS:
        st_val = _clip((STOCH_OS - stoch_k) / STOCH_OS + 0.6)
    else:
        st_val = _clip((stoch_k - 50) / 50.0) * 0.4
    add("Stoch RSI", st_val, f"%K {stoch_k:.1f}")

    # 11) 거래량 z-score: 방향은 가격 변화 부호를 따른다
    ret = float(c["RET"]) if pd.notna(c["RET"]) else 0.0
    vol_val = _clip(np.sign(ret) * min(abs(vol_z) / VOL_Z_HOT, 1.0))
    add("거래량 z", vol_val, f"z {vol_z:+.2f} ({'급증' if vol_z >= VOL_Z_HOT else '보통'})")

    # 12) 상위 시간프레임 정합: 상위 추세와 같은 방향일 때만 가점
    add("상위TF 정합", _clip(float(htf)), "상위TF 상승" if htf > 0 else ("상위TF 하락" if htf < 0 else "상위TF 혼조"))

    total_weight = sum(f["가중치"] for f in factors) or 1
    total = sum(f["기여도"] for f in factors) / total_weight * 100.0

    # 신뢰도: 데이터 충분성 + 국면 명확성 + 지표 유효성
    conf = 100.0
    conf -= 25 if len(df) < MIN_BARS_FOR_SIGNAL * 2 else 0
    conf -= 20 if regime == REGIME_VOLATILE else 0
    conf -= 10 if regime == REGIME_RANGE else 0
    conf -= 10 if htf == 0 else 0
    conf -= 15 if pd.isna(c["ADX"]) else 0
    confidence = float(max(conf, 10.0))
    return factors, float(total), confidence


def grade_signal(score: float, confidence: float, risk_gate: bool) -> tuple[str, str, str]:
    """(등급, 색상, 설명). 리스크 게이트를 통과하지 못하면 등급을 강제 하향한다."""
    if risk_gate:
        return "⛔ 진입 보류", "#8e44ad", "리스크 게이트 위반 — 신호와 무관하게 신규 진입 금지"
    if score >= SCORE_STRONG:
        return "🟢 강력 매수", "#1e8449", f"복합점수 {score:+.1f} / 신뢰도 {confidence:.0f}%"
    if score >= SCORE_MILD:
        return "🟩 매수 우위", "#28b463", f"복합점수 {score:+.1f} / 신뢰도 {confidence:.0f}%"
    if score <= -SCORE_STRONG:
        return "🔴 강력 매도", "#922b21", f"복합점수 {score:+.1f} / 신뢰도 {confidence:.0f}%"
    if score <= -SCORE_MILD:
        return "🟥 매도 우위", "#cb4335", f"복합점수 {score:+.1f} / 신뢰도 {confidence:.0f}%"
    return "⚪ 관망", "#7f8c8d", f"복합점수 {score:+.1f} / 신뢰도 {confidence:.0f}%"


def risk_gate(latest: pd.Series, atr_rank: float, regime: str) -> tuple[bool, list[str]]:
    """신규 진입을 막아야 하는 조건. (차단여부, 사유목록)."""
    reasons: list[str] = []
    atr_pct = float(latest.get("ATR_PCT") or 0.0)
    vol_z = float(latest.get("VOL_Z") or 0.0)
    if regime == REGIME_VOLATILE:
        reasons.append("방향성 없는 급변동 국면")
    if atr_pct >= 3.0:
        reasons.append(f"ATR {atr_pct:.2f}% — 1봉 변동성 과다")
    if abs(vol_z) >= VOL_Z_EXTREME:
        reasons.append(f"거래량 z {vol_z:+.2f} — 뉴스성 이상 거래")
    return (len(reasons) > 0), reasons


def atr_plan(latest: pd.Series, direction: int) -> dict:
    """ATR 기반 손절/목표/손익비. direction: +1 롱, -1 숏."""
    close = float(latest["close"])
    atr_v = float(latest["ATR"]) if pd.notna(latest["ATR"]) else 0.0
    stop = close - direction * ATR_STOP_MULT * atr_v
    target = close + direction * ATR_TARGET_MULT * atr_v
    risk = abs(close - stop)
    reward = abs(target - close)
    return {
        "기준가": close, "ATR": atr_v,
        "손절가": stop, "목표가": target,
        "손절폭%": (risk / close * 100) if close else 0.0,
        "목표폭%": (reward / close * 100) if close else 0.0,
        "손익비": (reward / risk) if risk > 0 else 0.0,
    }


# =============================================================================
# 6. 차트 패턴 — 프랙탈 피벗 + 허용 오차 기반 (원본의 5봉 단순 부등식 대체)
# =============================================================================
def find_pivots(high: np.ndarray, low: np.ndarray,
                left: int = PIVOT_LEFT, right: int = PIVOT_RIGHT):
    """좌우 left/right 봉보다 높은(낮은) 지점을 피벗으로 추출."""
    ph, pl = [], []
    for i in range(left, len(high) - right):
        window_h = high[i - left: i + right + 1]
        window_l = low[i - left: i + right + 1]
        if high[i] == window_h.max():
            ph.append(i)
        if low[i] == window_l.min():
            pl.append(i)
    return ph, pl


def _near(a: float, b: float, tol: float = PIVOT_TOL) -> bool:
    return abs(a - b) / max(abs(a), abs(b), 1e-12) <= tol


def detect_patterns(df: pd.DataFrame, lookback: int = 90) -> list[dict]:
    """허용 오차(PIVOT_TOL)와 최소 스윙 폭(MIN_SWING)으로 패턴을 판정한다.

    원본은 값이 정확히 부등호를 만족해야만 잡히는 구조라 실전에서 거의 발현되지 않았다.
    """
    out: list[dict] = []
    if len(df) < 30:
        return out
    seg = df.iloc[-lookback:]
    high = seg["high"].to_numpy(dtype=float)
    low = seg["low"].to_numpy(dtype=float)
    close = float(seg["close"].iloc[-1])
    ph, pl = find_pivots(high, low)
    hi_prices = [high[i] for i in ph]
    lo_prices = [low[i] for i in pl]

    # 이중 바닥 / 이중 천장
    if len(lo_prices) >= 2:
        a, b = lo_prices[-2], lo_prices[-1]
        mid = max(high[pl[-2]:pl[-1] + 1]) if pl[-1] > pl[-2] else max(high[pl[-1]:pl[-2] + 1])
        if _near(a, b) and (mid - max(a, b)) / max(a, b) >= MIN_SWING:
            out.append({"패턴": "이중 바닥(W)", "방향": 1,
                        "설명": f"저점 {a:,.1f} ≈ {b:,.1f}, 넥라인 {mid:,.1f}",
                        "목표": mid + (mid - max(a, b))})
    if len(hi_prices) >= 2:
        a, b = hi_prices[-2], hi_prices[-1]
        mid = min(low[ph[-2]:ph[-1] + 1]) if ph[-1] > ph[-2] else min(low[ph[-1]:ph[-2] + 1])
        if _near(a, b) and (min(a, b) - mid) / max(a, b) >= MIN_SWING:
            out.append({"패턴": "이중 천장(M)", "방향": -1,
                        "설명": f"고점 {a:,.1f} ≈ {b:,.1f}, 넥라인 {mid:,.1f}",
                        "목표": mid - (min(a, b) - mid)})

    # 헤드앤숄더 / 역 헤드앤숄더
    if len(hi_prices) >= 3:
        l, m, r = hi_prices[-3], hi_prices[-2], hi_prices[-1]
        if m > l and m > r and _near(l, r) and (m - max(l, r)) / m >= MIN_SWING * 0.8:
            neck = min(low[ph[-3]:ph[-1] + 1])
            out.append({"패턴": "헤드앤숄더", "방향": -1,
                        "설명": f"어깨 {l:,.1f}/{r:,.1f}, 머리 {m:,.1f}",
                        "목표": neck - (m - neck)})
    if len(lo_prices) >= 3:
        l, m, r = lo_prices[-3], lo_prices[-2], lo_prices[-1]
        if m < l and m < r and _near(l, r) and (min(l, r) - m) / m >= MIN_SWING * 0.8:
            neck = max(high[pl[-3]:pl[-1] + 1])
            out.append({"패턴": "역 헤드앤숄더", "방향": 1,
                        "설명": f"어깨 {l:,.1f}/{r:,.1f}, 머리 {m:,.1f}",
                        "목표": neck + (neck - m)})

    # 삼각형(수렴) — 최근 고점 하락 + 저점 상승
    if len(hi_prices) >= 2 and len(lo_prices) >= 2:
        if hi_prices[-1] < hi_prices[-2] and lo_prices[-1] > lo_prices[-2]:
            out.append({"패턴": "수렴 삼각형", "방향": 0,
                        "설명": "고점 하락 + 저점 상승 → 방향성 돌파 대기", "목표": np.nan})
        elif _near(hi_prices[-1], hi_prices[-2], PIVOT_TOL * 1.5) and lo_prices[-1] > lo_prices[-2]:
            out.append({"패턴": "상승 삼각형", "방향": 1,
                        "설명": "수평 저항 + 저점 상승 → 상방 돌파 우위", "목표": hi_prices[-1] * 1.03})
        elif _near(lo_prices[-1], lo_prices[-2], PIVOT_TOL * 1.5) and hi_prices[-1] < hi_prices[-2]:
            out.append({"패턴": "하락 삼각형", "방향": -1,
                        "설명": "수평 지지 + 고점 하락 → 하방 이탈 우위", "목표": lo_prices[-1] * 0.97})

    # 레인지 돌파 — 최근 구간 고/저 대비 현재가
    recent_high = float(seg["high"].iloc[:-1].max())
    recent_low = float(seg["low"].iloc[:-1].min())
    if recent_low > 0 and close >= recent_high * (1 - PIVOT_TOL / 2):
        out.append({"패턴": "저항 돌파 시도", "방향": 1,
                    "설명": f"구간 고점 {recent_high:,.1f} 근접/돌파", "목표": recent_high * 1.02})
    elif recent_high > 0 and close <= recent_low * (1 + PIVOT_TOL / 2):
        out.append({"패턴": "지지 이탈 시도", "방향": -1,
                    "설명": f"구간 저점 {recent_low:,.1f} 근접/이탈", "목표": recent_low * 0.98})

    return out


# =============================================================================
# 7. 리스크 관리 — 신호와 동급으로 취급
# =============================================================================
def volatility_metrics(df: pd.DataFrame, unit: int, window: int = RISK_WINDOW) -> dict:
    """연율 변동성, 1일 변동성, 최대낙폭(MDD), 하방편차."""
    ret = df["RET"].dropna().iloc[-window:]
    if len(ret) < 10:
        return {"연율변동성%": np.nan, "일변동성%": np.nan, "MDD%": np.nan, "하방변동성%": np.nan}
    sigma = float(ret.std(ddof=0))
    bars_per_day = 1440.0 / max(unit, 1)
    daily = sigma * np.sqrt(bars_per_day) * 100.0
    annual = sigma * np.sqrt(ANN_MINUTES / max(unit, 1)) * 100.0
    downside = ret[ret < 0]
    down_sigma = float(downside.std(ddof=0)) * np.sqrt(bars_per_day) * 100.0 if len(downside) > 2 else np.nan
    mdd = float(df["DRAWDOWN"].iloc[-window:].min()) * 100.0
    return {"연율변동성%": annual, "일변동성%": daily, "MDD%": mdd, "하방변동성%": down_sigma}


def position_size(portfolio_krw: float, entry: float, stop: float) -> dict:
    """고정비율 리스크(1% 룰) 기반 권장 수량/금액/비중."""
    risk_budget = portfolio_krw * RISK_PER_TRADE
    stop_dist = abs(entry - stop)
    if stop_dist <= 0 or entry <= 0 or portfolio_krw <= 0:
        return {"리스크예산": risk_budget, "손절폭": stop_dist, "권장수량": 0.0,
                "권장금액": 0.0, "권장비중%": 0.0}
    qty = risk_budget / stop_dist
    amount = qty * entry
    return {
        "리스크예산": risk_budget, "손절폭": stop_dist,
        "권장수량": qty, "권장금액": amount,
        "권장비중%": amount / portfolio_krw * 100.0,
    }


def parametric_var(daily_sigma_pct: float, value_krw: float, z: float = 1.645) -> float:
    """95% 신뢰수준 1일 파라메트릭 VaR(원)."""
    if np.isnan(daily_sigma_pct) or value_krw <= 0:
        return 0.0
    return z * daily_sigma_pct / 100.0 * value_krw


def concentration_flag(weight: float) -> tuple[str, str]:
    """단일 자산 비중 경고등. (등급, 색상)."""
    if weight >= CONCENTRATION_HIGH:
        return "심각", "#922b21"
    if weight >= CONCENTRATION_WARN:
        return "경고", "#b9770e"
    return "양호", "#1e8449"


def investment_committee(score: float, confidence: float, regime: str, gate: bool,
                         gate_reasons: list[str], rr: float, htf: int,
                         weight: float, mdd: float, latest: pd.Series) -> tuple[list[dict], str, str]:
    """냉철한 판단용 체크리스트. (항목별 결과, 종합판정, 색상) 반환."""
    checks: list[dict] = []

    def chk(name: str, ok: bool, detail: str, weight_note: str = "") -> None:
        checks.append({"심사항목": name, "판정": "통과" if ok else "미달",
                       "상세": detail, "비고": weight_note})

    chk("신호 강도", abs(score) >= SCORE_MILD, f"복합점수 {score:+.1f} (기준 ±{SCORE_MILD})",
        "강한 방향성 없으면 진입 금지")
    chk("신호 신뢰도", confidence >= 70, f"신뢰도 {confidence:.0f}% (기준 70%)", "지표 유효성·데이터 충분성")
    chk("추세 국면", regime in (REGIME_TREND, REGIME_STRONG_TREND), f"국면: {regime}", "횡보장은 기대값 낮음")
    chk("리스크 게이트", not gate, "위반 없음" if not gate else " / ".join(gate_reasons), "변동성·이상거래 차단")
    chk("손익비", rr >= MIN_RR, f"R:R {rr:.2f} (최소 {MIN_RR})", "기대값 양(+) 확보")
    chk("상위TF 정합", htf != 0, "상위 추세 일치" if htf else "상위 추세 혼조", "역추세 진입 회피")
    chk("집중도", weight < CONCENTRATION_WARN, f"포트폴리오 비중 {weight * 100:.1f}% (한도 {CONCENTRATION_WARN * 100:.0f}%)",
        "분산 원칙")
    chk("최대낙폭", mdd > -35.0, f"구간 MDD {mdd:.1f}%", "-35% 초과 시 구조 훼손")

    passed = sum(1 for c in checks if c["판정"] == "통과")
    total = len(checks)
    if passed == total and score >= SCORE_MILD:
        return checks, "✅ 진입 가결 (조건부)", "#1e8449"
    if passed >= total - 2 and abs(score) >= SCORE_MILD:
        return checks, "⚠️ 조건부 보류 — 분할 접근", "#b9770e"
    return checks, "🚫 부결 — 자본 보존 우선", "#922b21"


# =============================================================================
# 8. 알림 — 쿨다운/중복 억제 포함
# =============================================================================
def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram(message: str, force: bool = False) -> bool:
    """쿨다운(동일 메시지 TG_DEDUP_WINDOW 초) 내 재전송 금지. 전송 성공 여부 반환."""
    if not telegram_enabled():
        return False
    now = datetime.now()
    log = st.session_state.setdefault("telegram_sent_log", [])
    log[:] = [(m, t) for m, t in log if (now - t).total_seconds() < TG_DEDUP_WINDOW]
    if not force and any(m == message for m, _ in log):
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message[:4000],
                  "disable_web_page_preview": True},
            timeout=HTTP_TIMEOUT,
        )
        ok = resp.status_code == 200
        if ok:
            log.append((message, now))
            st.session_state.setdefault("alerts", []).append(
                f"[{now:%H:%M:%S}] 텔레그램 전송 · {message.splitlines()[0]}"
            )
        return ok
    except requests.RequestException as exc:
        st.session_state.setdefault("alerts", []).append(
            f"[{now:%H:%M:%S}] 텔레그램 전송 실패 · {exc.__class__.__name__}"
        )
        return False


def cooldown_ok(key: str, minutes: float) -> bool:
    """알림 종류별 쿨다운 통과 여부."""
    now = datetime.now()
    store = st.session_state.setdefault("alert_cooldown", {})
    if (now - store.get(key, datetime.min)) >= timedelta(minutes=minutes):
        store[key] = now
        return True
    return False


# =============================================================================
# 9. 세션 상태 / 로그인 (비밀번호는 st.secrets 에서만 읽는다)
# =============================================================================
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("logged_in", False)
    ss.setdefault("login_fails", 0)
    ss.setdefault("login_time", None)
    ss.setdefault("alerts", [])
    ss.setdefault("telegram_sent_log", [])
    ss.setdefault("alert_cooldown", {})
    ss.setdefault("api_error", None)
    ss.setdefault("api_fail_ts", None)
    ss.setdefault("last_table_alert", datetime.min)
    ss.setdefault("targets", {m: 0.0 for m in MARKETS})
    ss.setdefault("reentries", {m: 0.0 for m in MARKETS})
    ss.setdefault("goals", dict(DEFAULT_GOALS))
    ss.setdefault("price_editor_df", pd.DataFrame({
        "코인": [m.split("-")[1] for m in MARKETS],
        "목표가(매도)": [0.0] * len(MARKETS),
        "재진입가(매수)": [0.0] * len(MARKETS),
        "목표수량": [DEFAULT_GOALS[m] for m in MARKETS],
    }))


def _safe_eq(a, b) -> bool:
    """상수시간 비교. 비ASCII 입력에서도 예외로 죽지 않는다."""
    try:
        return hmac.compare_digest(str(a).encode("utf-8"), str(b).encode("utf-8"))
    except Exception:                              # noqa: BLE001
        return False


def check_login() -> bool:
    """세션 로그인. 비밀번호 미설정 시 앱을 열지 않고 안내만 표시한다."""
    ss = st.session_state
    if ss.logged_in:
        elapsed = datetime.now() - ss.login_time if ss.login_time else timedelta(0)
        if elapsed > timedelta(minutes=SESSION_TIMEOUT_MIN):
            ss.logged_in = False
            st.warning(f"세션 만료({SESSION_TIMEOUT_MIN}분) — 다시 로그인해 주세요.")
        else:
            return True

    if not APP_PW:
        st.error("### ⚠️ 초기 설정 필요\n"
                 "`APP_PW`(그리고 선택적으로 `APP_ID`)가 설정되지 않았습니다.\n\n"
                 "Streamlit Cloud → **Settings → Secrets** 에 아래를 붙여넣고 저장하세요.\n"
                 "```toml\nAPP_ID = \"miwooni\"\nAPP_PW = \"사용할_강한_비밀번호\"\n"
                 "TELEGRAM_BOT_TOKEN = \"봇토큰\"\nTELEGRAM_CHAT_ID = \"챗아이디\"\n```")
        st.stop()

    st.title("🔐 로그인 필요")
    _, mid, _ = st.columns([1, 2, 1])
    with mid, st.form("login_form", clear_on_submit=False):
        st.subheader("MIWOONI 대시보드")
        input_id = st.text_input("아이디")
        input_pw = st.text_input("비밀번호", type="password")
        if st.form_submit_button("로그인", use_container_width=True):
            id_ok = _safe_eq(input_id, APP_ID or "")
            pw_ok = _safe_eq(input_pw, APP_PW)
            if id_ok and pw_ok:
                ss.logged_in = True
                ss.login_time = datetime.now()
                ss.login_fails = 0
                st.rerun()
            ss.login_fails += 1
            time.sleep(LOGIN_FAIL_SLEEP)
            if ss.login_fails >= LOGIN_MAX_FAILS:
                st.error(f"로그인 실패 {ss.login_fails}회 — 잠시 후 다시 시도하세요.")
            else:
                st.error(f"아이디 또는 비밀번호가 올바르지 않습니다. (실패 {ss.login_fails}/{LOGIN_MAX_FAILS})")
    return False


# =============================================================================
# 10. 포트폴리오 평가
# =============================================================================
def build_portfolio(prices: dict) -> dict:
    """보유 평가액·비중·목표 달성률을 한 번에 계산한다(0 가격 방어 포함)."""
    rows, total = [], 0.0
    for m in MARKETS:
        price = parse_number(prices.get(m, {}).get("trade_price", 0.0))
        qty = HOLDINGS.get(m, 0.0)
        value = price * qty
        total += value
        rows.append({"market": m, "coin": m.split("-")[1], "price": price, "qty": qty, "value": value})

    for r in rows:
        r["weight"] = (r["value"] / total) if total > 0 else 0.0
        goal = float(st.session_state.goals.get(r["market"], 0.0) or 0.0)
        r["goal"] = goal
        r["goal_pct"] = (r["qty"] / goal * 100.0) if goal > 0 else np.nan
        # 목표 달성까지 필요한 추가 매수 금액(수수료 반영)
        need_qty = max(goal - r["qty"], 0.0)
        r["goal_cost"] = need_qty * r["price"] * (1 + UPBIT_FEE) if r["price"] > 0 else 0.0
    return {"rows": {r["market"]: r for r in rows}, "total": total}


# =============================================================================
# 11. 차트
# =============================================================================
PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False}


def build_chart(df: pd.DataFrame, coin: str, tf_label: str, target: float, reentry: float) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.52, 0.13, 0.17, 0.18],
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
        subplot_titles=("가격 · HMA · 볼린저 · VWAP", "거래량", "RSI(14) · Stoch RSI", "MACD · ADX"),
    )
    x = df["datetime"]
    fig.add_trace(go.Candlestick(x=x, open=df["open"], high=df["high"], low=df["low"],
                                 close=df["close"], name="가격",
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["BB_upper"], name="BB 상한", line=dict(color="#9aa0a6", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["BB_lower"], name="BB 하한", line=dict(color="#9aa0a6", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(154,160,166,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["HMA"], name=f"HMA{HMA_PERIOD}", line=dict(color="#1e88e5", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["HMA3"], name="HMA3", line=dict(color="#fb8c00", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["EMA55"], name="EMA55", line=dict(color="#8e24aa", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["VWAP"], name=f"VWAP{VWAP_WINDOW}", line=dict(color="#00acc1", width=1.2)), row=1, col=1)

    last = df.iloc[-1]
    fig.add_hline(y=float(last["close"]), line_dash="dash", line_color="#fdd835", row=1, col=1,
                  annotation_text=f"현재 {float(last['close']):,.1f}", annotation_position="top left")
    if target > 0:
        fig.add_hline(y=target, line_color="#e53935", line_width=1.4, row=1, col=1,
                      annotation_text=f"목표 {target:,.1f}", annotation_position="bottom left")
    if reentry > 0:
        fig.add_hline(y=reentry, line_color="#1e88e5", line_width=1.4, row=1, col=1,
                      annotation_text=f"재진입 {reentry:,.1f}", annotation_position="bottom left")
    stop = float(last["close"]) - ATR_STOP_MULT * float(last["ATR"]) if pd.notna(last["ATR"]) else 0.0
    if stop > 0:
        fig.add_hline(y=stop, line_color="#ef6c00", line_width=1.1, line_dash="dot", row=1, col=1,
                      annotation_text=f"ATR 손절 {stop:,.1f}", annotation_position="bottom right")

    fig.add_trace(go.Bar(x=x, y=df["volume"], name="거래량",
                         marker_color=np.where(df["close"] >= df["open"], "#26a69a", "#ef5350")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["VOL_MA"], name="거래량 MA", line=dict(color="#fdd835", width=1)), row=2, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["RSI"], name=f"RSI({RSI_PERIOD})", line=dict(color="#8e24aa", width=1.4)), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["STOCH_K"], name="Stoch %K", line=dict(color="#00acc1", width=1, dash="dot")), row=3, col=1)
    for lvl, col in ((RSI_OB, "#e53935"), (RSI_OS, "#43a047"), (RSI_TREND_PIVOT, "#9e9e9e")):
        fig.add_hline(y=lvl, line_dash="dot", line_color=col, line_width=1, row=3, col=1)

    fig.add_trace(go.Bar(x=x, y=df["MACD_hist"], name="MACD 히스토그램",
                         marker_color=np.where(df["MACD_hist"] >= 0, "#26a69a", "#ef5350")), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["MACD_line"], name="MACD", line=dict(color="#1e88e5", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["Signal_line"], name="Signal", line=dict(color="#fb8c00", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["ADX"], name=f"ADX({ADX_PERIOD})",
                             line=dict(color="#e0e0e0", width=1.4)), row=4, col=1, secondary_y=True)
    # 기준선은 반드시 보조축(secondary_y)에 그린다 — 주축에 그리면 MACD 스케일이 붕괴한다
    fig.add_hline(y=ADX_TREND_MIN, line_dash="dot", line_color="#9e9e9e", line_width=1,
                  row=4, col=1, secondary_y=True)
    fig.add_hline(y=ADX_STRONG, line_dash="dot", line_color="#616161", line_width=1,
                  row=4, col=1, secondary_y=True)

    macd_abs = float(np.nanmax(np.abs(df["MACD_hist"].to_numpy(dtype=float))))
    if not np.isfinite(macd_abs) or macd_abs <= 0:
        macd_abs = 1.0
    fig.update_yaxes(range=[-macd_abs * 2.6, macd_abs * 2.6], row=4, col=1, secondary_y=False)
    fig.update_yaxes(range=[0, 60], showgrid=False, row=4, col=1, secondary_y=True)

    fig.update_layout(
        height=820, title=f"📌 {coin} · {tf_label}봉", xaxis_rangeslider_visible=False,
        margin=dict(t=70, b=30, l=10, r=10), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def build_gauge(score: float, confidence: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(score, 1),
        number={"suffix": " 점", "font": {"size": 30}},
        title={"text": f"복합 신호 점수 (신뢰도 {confidence:.0f}%)", "font": {"size": 13}},
        gauge={
            "axis": {"range": [-100, 100], "tickwidth": 1},
            "bar": {"color": "#37474f"},
            "steps": [
                {"range": [-100, -SCORE_STRONG], "color": "#922b21"},
                {"range": [-SCORE_STRONG, -SCORE_MILD], "color": "#cb4335"},
                {"range": [-SCORE_MILD, SCORE_MILD], "color": "#7f8c8d"},
                {"range": [SCORE_MILD, SCORE_STRONG], "color": "#28b463"},
                {"range": [SCORE_STRONG, 100], "color": "#1e8449"},
            ],
            "threshold": {"line": {"color": "#fdd835", "width": 3}, "value": score},
        },
    ))
    fig.update_layout(height=210, margin=dict(t=45, b=5, l=15, r=15),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# =============================================================================
# 12. 메인 애플리케이션
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="MIWOONI Trading Dashboard", page_icon="🔥",
                       layout="wide", initial_sidebar_state="expanded")
    init_state()
    if not check_login():
        st.stop()

    if HAS_AUTOREFRESH:
        st_autorefresh(interval=REFRESH_MS, key="auto_refresh")

    # ---------------- 사이드바 ----------------
    with st.sidebar:
        st.header("⚙️ 제어 패널")
        selected_tf = st.selectbox("차트 주기", list(TIMEFRAMES.keys()),
                                   format_func=lambda k: TIMEFRAMES[k], index=4)
        htf_unit = HTF_MAP.get(selected_tf, selected_tf)
        st.caption(f"상위 시간프레임 자동 대조: {TIMEFRAMES[htf_unit]}봉")

        if st.session_state.login_time:
            elapsed = datetime.now() - st.session_state.login_time
            st.info(f"로그인 경과 {elapsed.seconds // 3600}시간 {(elapsed.seconds % 3600) // 60}분 "
                    f"/ 만료까지 {SESSION_TIMEOUT_MIN - int(elapsed.total_seconds() // 60)}분")
        c1, c2 = st.columns(2)
        if c1.button("🔄 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        if c2.button("🚪 로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

        telegram_on = st.checkbox("🔔 텔레그램 알림", value=True,
                                  disabled=not telegram_enabled(),
                                  help="secrets에 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 필요")
        if not telegram_enabled():
            st.caption("⚠️ 텔레그램 미설정 — 알림은 화면 로그에만 기록됩니다.")
        table_alert_min = st.number_input("요약 리포트 주기(분)", 1, 240, 30, 5)

        st.markdown("---")
        st.subheader("🎯 가격 · 수량 목표")
        st.caption("표를 직접 편집하세요. 0 = 미설정")
        edited = st.data_editor(
            st.session_state.price_editor_df, num_rows="fixed", hide_index=True,
            use_container_width=True, key="price_editor",
            column_config={
                "코인": st.column_config.TextColumn("코인", disabled=True, width="small"),
                "목표가(매도)": st.column_config.NumberColumn("목표가(매도)", min_value=0.0, step=10.0, format="%.1f"),
                "재진입가(매수)": st.column_config.NumberColumn("재진입가(매수)", min_value=0.0, step=10.0, format="%.1f"),
                "목표수량": st.column_config.NumberColumn("목표수량(개)", min_value=0.0, step=100.0, format="%.0f"),
            },
        )
        st.session_state.price_editor_df = edited
        for i, m in enumerate(MARKETS):
            st.session_state.targets[m] = parse_number(edited.iloc[i]["목표가(매도)"])
            st.session_state.reentries[m] = parse_number(edited.iloc[i]["재진입가(매수)"])
            st.session_state.goals[m] = parse_number(edited.iloc[i]["목표수량"])

        with st.expander("📐 리스크 파라미터(고정값)"):
            st.write(
                f"- 1회 매매 리스크: 자산의 **{RISK_PER_TRADE * 100:.0f}%** (고정비율)\n"
                f"- 손절: **{ATR_STOP_MULT} × ATR**, 목표: **{ATR_TARGET_MULT} × ATR** → 최소 R:R {ATR_TARGET_MULT / ATR_STOP_MULT:.1f}\n"
                f"- 추세 판별: ADX ≥ **{ADX_TREND_MIN}** (강추세 ≥ {ADX_STRONG})\n"
                f"- 과매수/과매도: RSI **{RSI_OB}/{RSI_OS}**, Stoch RSI **{STOCH_OB}/{STOCH_OS}**\n"
                f"- 거래량 급증: z ≥ **{VOL_Z_HOT}**, 이상 급증 ≥ **{VOL_Z_EXTREME}**\n"
                f"- 집중도 경고: 단일 자산 **{CONCENTRATION_WARN * 100:.0f}%** 초과\n"
                f"- 수수료: 왕복 **{UPBIT_FEE * 200:.2f}%** 반영"
            )

    # ---------------- 헤더 ----------------
    st.title("🔥 If you're not desperate, don't even think about it!")
    st.markdown("#### 📈 실시간 암호화폐 수량 증식 · 국면 인식형 파동 분석 대시보드")

    prices, err = get_prices()
    if err:
        st.error(f"⚠️ 업비트 API 오류 — 데이터가 갱신되지 않았습니다.\n\n`{err}`\n\n"
                 "· Streamlit Cloud의 해외 IP가 업비트에서 **지역 차단(HTTP 403/418)** 될 수 있습니다. "
                 "이 경우 별도 프록시/프라이빗 서버를 통한 데이터 수집이 필요합니다.\n"
                 "· 최대 60초 후 자동 재시도합니다.")
    if not prices:
        st.warning("표시할 시세가 없습니다. 잠시 후 새로고침하거나 사이드바의 🔄 버튼을 눌러주세요.")
        st.stop()

    portfolio = build_portfolio(prices)
    rows = portfolio["rows"]
    total_value = portfolio["total"]

    # ---------------- 분석 루프(1회 순회로 모든 지표 산출) ----------------
    analysis: dict[str, dict] = {}
    for m in MARKETS:
        df = load_ohlcv(m, selected_tf)
        if df.empty or len(df) < MIN_BARS_FOR_SIGNAL:
            continue
        last = df.iloc[-1]
        atr_rank = float(df["ATR_PCT"].rolling(100).rank(pct=True).iloc[-1] * 100) if len(df) >= 100 else 50.0
        regime, regime_reason = classify_regime(last, atr_rank)
        htf = htf_trend(m, selected_tf)
        gate, gate_reasons = risk_gate(last, atr_rank, regime)
        factors, score, conf = score_factors(df, htf, regime)
        direction = 1 if score > 0 else -1
        plan = atr_plan(last, direction)
        grade, grade_color, grade_note = grade_signal(score, conf, gate)
        vol = volatility_metrics(df, selected_tf)
        patterns = detect_patterns(df)
        r = rows[m]
        size = position_size(total_value, plan["기준가"], plan["손절가"])
        var95 = parametric_var(vol["일변동성%"], r["value"])
        analysis[m] = {
            "df": df, "last": last, "regime": regime, "regime_reason": regime_reason,
            "htf": htf, "gate": gate, "gate_reasons": gate_reasons,
            "factors": factors, "score": score, "conf": conf, "plan": plan,
            "grade": grade, "grade_color": grade_color, "grade_note": grade_note,
            "vol": vol, "patterns": patterns, "size": size, "var95": var95,
            "atr_rank": atr_rank,
        }

    # ---------------- 포트폴리오 KPI ----------------
    sand = rows[BASE_MARKET]
    pnl = sand["value"] - TOTAL_INVESTMENT
    pnl_pct = (pnl / TOTAL_INVESTMENT * 100.0) if TOTAL_INVESTMENT else 0.0
    port_daily_vol = np.sqrt(sum((rows[m]["weight"] * (analysis[m]["vol"]["일변동성%"] if m in analysis else 0.0)) ** 2
                                 for m in MARKETS)) if analysis else np.nan
    port_var = parametric_var(port_daily_vol, total_value)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("총 매수 원금", f"{TOTAL_INVESTMENT:,.0f} 원")
    k2.metric("포트폴리오 평가액", f"{total_value:,.0f} 원",
              f"{total_value - TOTAL_INVESTMENT:+,.0f} 원")
    k3.metric(f"{sand['coin']} 평가액", f"{sand['value']:,.0f} 원", f"{pnl:+,.0f} 원 ({pnl_pct:+.2f}%)")
    k4.metric("포트 일변동성(σ)", f"{port_daily_vol:.2f} %" if not np.isnan(port_daily_vol) else "-",
              help="보유 비중으로 가중한 일간 변동성")
    k5.metric("1일 VaR 95%", f"{port_var:,.0f} 원", help="95% 신뢰수준 1일 최대 예상 손실(파라메트릭)")

    # ---------------- 종합 테이블 ----------------
    st.markdown("---")
    st.subheader(f"📊 종합 판단 테이블 (기축 {sand['coin']} · {TIMEFRAMES[selected_tf]}봉)")
    table_rows = []
    for m in MARKETS:
        r = rows[m]
        a = analysis.get(m)
        t_price, r_price = st.session_state.targets[m], st.session_state.reentries[m]
        row = {
            "코인": r["coin"], "현재가": r["price"], "등락률%": parse_number(prices.get(m, {}).get("signed_change_rate", 0)) * 100,
            "보유수량": r["qty"], "평가액": r["value"], "비중%": r["weight"] * 100,
            "목표가": t_price, "재진입가": r_price, "목표수량": r["goal"], "달성률%": r["goal_pct"],
        }
        if a:
            row.update({
                "국면": a["regime"], "복합점수": a["score"], "신뢰도%": a["conf"], "판정": a["grade"],
                "RSI": float(a["last"]["RSI"]) if pd.notna(a["last"]["RSI"]) else np.nan,
                "ADX": float(a["last"]["ADX"]) if pd.notna(a["last"]["ADX"]) else np.nan,
                "ATR%": float(a["last"]["ATR_PCT"]) if pd.notna(a["last"]["ATR_PCT"]) else np.nan,
                "거래량z": float(a["last"]["VOL_Z"]) if pd.notna(a["last"]["VOL_Z"]) else np.nan,
                "손절가": a["plan"]["손절가"], "목표가(ATR)": a["plan"]["목표가"], "손익비": a["plan"]["손익비"],
                "일변동성%": a["vol"]["일변동성%"], "MDD%": a["vol"]["MDD%"], "패턴": ", ".join(p["패턴"] for p in a["patterns"]) or "-",
            })
        table_rows.append(row)
    tdf = pd.DataFrame(table_rows)
    if "달성률%" in tdf.columns:          # ProgressColumn 은 NaN 을 렌더링하지 못한다
        tdf["달성률%"] = tdf["달성률%"].apply(lambda v: None if pd.isna(v) else float(v))

    st.dataframe(
        tdf, hide_index=True, use_container_width=True,
        column_config={
            "현재가": st.column_config.NumberColumn(format="%.2f"),
            "등락률%": st.column_config.NumberColumn(format="%+.2f%%"),
            "보유수량": st.column_config.NumberColumn(format="%.2f"),
            "평가액": st.column_config.NumberColumn(format="%.0f"),
            "비중%": st.column_config.NumberColumn(format="%.2f%%"),
            "목표가": st.column_config.NumberColumn(format="%.1f"),
            "재진입가": st.column_config.NumberColumn(format="%.1f"),
            "목표수량": st.column_config.NumberColumn(format="%.0f"),
            "달성률%": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "복합점수": st.column_config.NumberColumn(format="%+.1f"),
            "신뢰도%": st.column_config.NumberColumn(format="%.0f%%"),
            "RSI": st.column_config.NumberColumn(format="%.1f"),
            "ADX": st.column_config.NumberColumn(format="%.1f"),
            "ATR%": st.column_config.NumberColumn(format="%.2f %%"),
            "거래량z": st.column_config.NumberColumn(format="%+.2f"),
            "손절가": st.column_config.NumberColumn(format="%.2f"),
            "목표가(ATR)": st.column_config.NumberColumn(format="%.2f"),
            "손익비": st.column_config.NumberColumn(format="%.2f"),
            "일변동성%": st.column_config.NumberColumn(format="%.2f %%"),
            "MDD%": st.column_config.NumberColumn(format="%.1f %%"),
        },
    )

    # ---------------- 대체 매수 시뮬레이션(수수료 1회만 반영) ----------------
    with st.expander("🔁 기축 자산 → 대체 코인 교체 시뮬레이션 (수수료 왕복 반영)"):
        st.caption(f"기축 {sand['coin']} 평가액에서 매도수수료 {UPBIT_FEE * 100:.2f}%를 차감한 뒤, "
                   f"대상 코인 매수수수료 {UPBIT_FEE * 100:.2f}%를 반영해 수량을 산출합니다.")
        sand_net = sand["value"] * (1 - UPBIT_FEE)
        sim = []
        for m in MARKETS:
            if m == BASE_MARKET:
                continue
            r = rows[m]
            if r["price"] <= 0:
                continue
            qty = sand_net * (1 - UPBIT_FEE) / r["price"]
            sim.append({
                "코인": r["coin"], "현재가": r["price"], "현재 보유": r["qty"],
                "교체 시 수량": qty, "수량 차이": qty - r["qty"],
                "교체 후 평가액": qty * r["price"], "증감액": qty * r["price"] - r["value"],
            })
        st.dataframe(pd.DataFrame(sim), hide_index=True, use_container_width=True,
                     column_config={"현재가": st.column_config.NumberColumn(format="%.2f"),
                                    "현재 보유": st.column_config.NumberColumn(format="%.2f"),
                                    "교체 시 수량": st.column_config.NumberColumn(format="%.2f"),
                                    "수량 차이": st.column_config.NumberColumn(format="%+.2f"),
                                    "교체 후 평가액": st.column_config.NumberColumn(format="%.0f"),
                                    "증감액": st.column_config.NumberColumn(format="%+.0f")})

    # ---------------- 투자심사위원회 ----------------
    st.markdown("---")
    st.subheader("🧮 투자심사 체크리스트 — 감정 배제, 조건 충족 여부만 판정")
    cands = [m for m in MARKETS if m in analysis and rows[m]["qty"] > 0] or [m for m in MARKETS if m in analysis]
    if not cands:
        st.info("분석 가능한 데이터가 없습니다.")
    else:
        tabs = st.tabs([f"{rows[m]['coin']} ({analysis[m]['grade']})" for m in cands])
        for tab, m in zip(tabs, cands):
            a, r = analysis[m], rows[m]
            with tab:
                g1, g2 = st.columns([1, 2])
                with g1:
                    st.plotly_chart(build_gauge(a["score"], a["conf"]), use_container_width=True,
                                    config=PLOT_CONFIG, key=f"gauge_{m}")
                with g2:
                    st.markdown(
                        f"**판정: <span style='color:{a['grade_color']}'>{a['grade']}</span>** "
                        f"<span style='color:#9e9e9e'>· {a['grade_note']}</span>",
                        unsafe_allow_html=True)
                    st.markdown(f"- **국면**: {a['regime']} — {a['regime_reason']}")
                    st.markdown(f"- **상위TF({TIMEFRAMES[HTF_MAP.get(selected_tf, selected_tf)]})**: "
                                f"{'상승' if a['htf'] > 0 else ('하락' if a['htf'] < 0 else '혼조')}")
                    if a["gate"]:
                        st.error("⛔ 리스크 게이트: " + " / ".join(a["gate_reasons"]))
                    p = a["plan"]
                    st.markdown(
                        f"- **ATR 기반 계획**: 기준 {p['기준가']:,.2f} / 손절 {p['손절가']:,.2f}({p['손절폭%']:.2f}%) "
                        f"/ 목표 {p['목표가']:,.2f}({p['목표폭%']:.2f}%) / **R:R {p['손익비']:.2f}**")
                    s = a["size"]
                    st.markdown(
                        f"- **권장 포지션(1% 룰)**: {s['권장수량']:,.2f}개 ≈ {s['권장금액']:,.0f}원 "
                        f"(포트 {s['권장비중%']:.1f}%, 리스크 예산 {s['리스크예산']:,.0f}원)")
                    v = a["vol"]
                    st.markdown(
                        f"- **리스크 지표**: 일변동성 {v['일변동성%']:.2f}% / 연율 {v['연율변동성%']:.1f}% "
                        f"/ 구간 MDD {v['MDD%']:.1f}% / 1일 VaR(95%) {a['var95']:,.0f}원")

                checks, verdict, vcolor = investment_committee(
                    a["score"], a["conf"], a["regime"], a["gate"], a["gate_reasons"],
                    a["plan"]["손익비"], a["htf"], r["weight"], a["vol"]["MDD%"], a["last"])
                st.markdown(f"#### 종합: <span style='color:{vcolor}'>{verdict}</span>", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(checks), hide_index=True, use_container_width=True)

                fc = pd.DataFrame(a["factors"])
                fc["기여도"] = fc["기여도"].round(2)
                st.markdown("##### 팩터별 기여도 (가중치 × 정규화 판독값)")
                st.dataframe(fc, hide_index=True, use_container_width=True,
                             column_config={"가중치": st.column_config.NumberColumn(format="%.0f"),
                                            "판독값": st.column_config.NumberColumn(format="%+.3f"),
                                            "기여도": st.column_config.NumberColumn(format="%+.2f")})
                if a["patterns"]:
                    st.markdown("##### 감지된 차트 패턴")
                    st.dataframe(pd.DataFrame(a["patterns"]), hide_index=True, use_container_width=True)
                else:
                    st.caption("감지된 패턴 없음")

    # ---------------- 상세 차트 ----------------
    st.markdown("---")
    st.subheader("📈 개별 코인 상세 차트")
    for m in MARKETS:
        if m not in analysis:
            st.warning(f"{rows[m]['coin']}: 데이터 부족으로 차트를 표시할 수 없습니다.")
            continue
        a = analysis[m]
        st.plotly_chart(build_chart(a["df"], rows[m]["coin"], TIMEFRAMES[selected_tf],
                                    st.session_state.targets[m], st.session_state.reentries[m]),
                        use_container_width=True, config=PLOT_CONFIG, key=f"chart_{m}")

    # ---------------- 알림 처리 ----------------
    now = datetime.now()
    for m in MARKETS:
        if m not in analysis:
            continue
        a, r = analysis[m], rows[m]
        coin, price = r["coin"], r["price"]
        last = a["last"]

        if st.session_state.targets[m] > 0 and price >= st.session_state.targets[m] and cooldown_ok(f"{coin}_target", ALERT_COOLDOWN["target"]):
            msg = (f"🎯 [{coin}] 목표가 도달\n현재 {price:,.2f}원 ≥ 목표 {st.session_state.targets[m]:,.2f}원\n"
                   f"판정 {a['grade']} · RSI {float(last['RSI']):.1f} · ADX {float(last['ADX']):.1f}\n"
                   f"ATR 손절 {a['plan']['손절가']:,.2f} / 목표 {a['plan']['목표가']:,.2f} (R:R {a['plan']['손익비']:.2f})")
            if telegram_on:
                send_telegram(msg, force=True)
            st.session_state.alerts.append(f"[{now:%H:%M:%S}] 🎯 {coin} 목표가 도달 ({price:,.2f}원)")

        if st.session_state.reentries[m] > 0 and price <= st.session_state.reentries[m] and cooldown_ok(f"{coin}_reentry", ALERT_COOLDOWN["reentry"]):
            msg = (f"🔄 [{coin}] 재진입가 도달\n현재 {price:,.2f}원 ≤ 재진입 {st.session_state.reentries[m]:,.2f}원\n"
                   f"판정 {a['grade']} · 국면 {a['regime']}\n권장수량 {a['size']['권장수량']:,.2f}개 "
                   f"(리스크 예산 {a['size']['리스크예산']:,.0f}원)")
            if telegram_on:
                send_telegram(msg, force=True)
            st.session_state.alerts.append(f"[{now:%H:%M:%S}] 🔄 {coin} 재진입가 도달 ({price:,.2f}원)")

        rsi_v = float(last["RSI"]) if pd.notna(last["RSI"]) else 50.0
        if rsi_v >= RSI_OB and cooldown_ok(f"{coin}_rsi_ob", ALERT_COOLDOWN["rsi_ob"]):
            if telegram_on:
                send_telegram(f"⚠️ [{coin}] RSI 과매수 {rsi_v:.1f} (≥{RSI_OB}) · 현재 {price:,.2f}원 · 국면 {a['regime']}")
            st.session_state.alerts.append(f"[{now:%H:%M:%S}] ⚠️ {coin} RSI 과매수 {rsi_v:.1f}")
        if rsi_v <= RSI_OS and cooldown_ok(f"{coin}_rsi_os", ALERT_COOLDOWN["rsi_os"]):
            if telegram_on:
                send_telegram(f"💡 [{coin}] RSI 과매도 {rsi_v:.1f} (≤{RSI_OS}) · 현재 {price:,.2f}원 · 국면 {a['regime']}")
            st.session_state.alerts.append(f"[{now:%H:%M:%S}] 💡 {coin} RSI 과매도 {rsi_v:.1f}")

        vz = float(last["VOL_Z"]) if pd.notna(last["VOL_Z"]) else 0.0
        if abs(vz) >= VOL_Z_EXTREME and cooldown_ok(f"{coin}_vol", ALERT_COOLDOWN["vol_spike"]):
            if telegram_on:
                send_telegram(f"🚨 [{coin}] 거래량 이상 급증 z={vz:+.2f} · 현재 {price:,.2f}원 · 뉴스/이벤트 확인 필요")
            st.session_state.alerts.append(f"[{now:%H:%M:%S}] 🚨 {coin} 거래량 이상 급증 z={vz:+.2f}")

        if a["score"] >= SCORE_STRONG and not a["gate"] and cooldown_ok(f"{coin}_buy", ALERT_COOLDOWN["grade"]):
            if telegram_on:
                send_telegram(f"🟢 [{coin}] 강력 매수 신호 {a['score']:+.1f} (신뢰도 {a['conf']:.0f}%)\n"
                              f"국면 {a['regime']} · 손절 {a['plan']['손절가']:,.2f} / 목표 {a['plan']['목표가']:,.2f} "
                              f"(R:R {a['plan']['손익비']:.2f})")
            st.session_state.alerts.append(f"[{now:%H:%M:%S}] 🟢 {coin} 강력 매수 {a['score']:+.1f}")
        if a["score"] <= -SCORE_STRONG and cooldown_ok(f"{coin}_sell", ALERT_COOLDOWN["grade"]):
            if telegram_on:
                send_telegram(f"🔴 [{coin}] 강력 매도 신호 {a['score']:+.1f} (신뢰도 {a['conf']:.0f}%)\n"
                              f"국면 {a['regime']} · 비중 {r['weight'] * 100:.1f}%")
            st.session_state.alerts.append(f"[{now:%H:%M:%S}] 🔴 {coin} 강력 매도 {a['score']:+.1f}")

        for pat in a["patterns"]:
            if cooldown_ok(f"{coin}_{pat['패턴']}", ALERT_COOLDOWN["pattern"]):
                tgt = f" · 예상목표 {pat['목표']:,.2f}원" if pd.notna(pat.get("목표")) else ""
                if telegram_on:
                    send_telegram(f"📐 [{coin}] 패턴 감지: {pat['패턴']} ({pat['설명']}){tgt}\n현재 {price:,.2f}원")
                st.session_state.alerts.append(f"[{now:%H:%M:%S}] 📐 {coin} {pat['패턴']}")

    # 정기 요약 리포트
    if (now - st.session_state.last_table_alert) > timedelta(minutes=float(table_alert_min)):
        lines = [f"📊 [MIWOONI] 정기 리포트 {now:%Y-%m-%d %H:%M}",
                 f"포트폴리오 {total_value:,.0f}원 · 일변동성 {port_daily_vol:.2f}% · VaR95 {port_var:,.0f}원", ""]
        for m in MARKETS:
            r = rows[m]
            if m in analysis:
                a = analysis[m]
                lines.append(f"• {r['coin']} {r['price']:,.2f}원 | {a['score']:+.1f}점 {a['grade']} | "
                             f"{a['regime']} | RSI {float(a['last']['RSI']):.0f} ADX {float(a['last']['ADX']):.0f} | "
                             f"비중 {r['weight'] * 100:.1f}%")
            else:
                lines.append(f"• {r['coin']} {r['price']:,.2f}원 | 데이터 부족")
        st.session_state.last_table_alert = now
        if telegram_on:
            send_telegram("\n".join(lines))
        st.session_state.alerts.append(f"[{now:%H:%M:%S}] 📊 정기 리포트 전송")

    # ---------------- 알림 로그 ----------------
    st.markdown("---")
    st.subheader("🔔 실시간 알림 로그")
    logs = st.session_state.alerts[-30:]
    if not logs:
        st.caption("아직 발생한 알림이 없습니다.")
    for entry in reversed(logs):
        bg = "#1a5276" if ("목표가 도달" in entry or "재진입가 도달" in entry) else (
            "#7d6608" if "🚨" in entry else "#212f3d")
        st.markdown(f"<div style='padding:8px 12px;background:{bg};border-radius:6px;"
                    f"margin:4px 0;font-size:0.88rem'>{entry}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("⚠️ 본 대시보드는 정보 제공 및 의사결정 보조 도구이며 투자 자문이 아닙니다. "
               "지표·신호는 과거 데이터의 통계적 요약일 뿐 미래 수익을 보장하지 않습니다. "
               "모든 매매 판단과 결과에 대한 책임은 사용자 본인에게 있습니다.")


if __name__ == "__main__":        # Streamlit 은 이 스크립트를 __main__ 으로 실행한다
    main()

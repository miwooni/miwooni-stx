# 통합-HBAR_최종.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import time
import re

# ---------------------- 공통 함수 정의 ----------------------
def send_telegram_alert(message: str):
    bot_token = "7545404316:AAHMdayWZjwEZmZwd5JrKXPDn5wUQfivpTw"
    chat_id = "7890657899"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        res = requests.post(url, data=payload, timeout=5)
        if res.status_code != 200:
            st.error(f"Telegram 전송 실패: {res.status_code} {res.text}")
        else:
            if 'alerts' in st.session_state:
                st.session_state.alerts.append(f"텔레그램 전송: {message}")
    except Exception as e:
        st.error(f"Telegram 알림 전송 실패: {str(e)}")

def parse_number(x):
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        if x.strip() == "-":
            return x
        cleaned = x.replace(',', '').strip()
        return float(cleaned) if cleaned else 0.0
    return 0.0
    
def wma(series, period):
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def hma(series, period=16):
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))
    wma_half = wma(series, half_length)
    wma_full = wma(series, period)
    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, sqrt_length)

def hma3(series, period=3):
    return hma(series, period)

def rsi(series, period=7):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def bollinger_bands(series, period=20, std_factor=2):
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + std_factor * std
    lower = ma - std_factor * std
    return ma, upper, lower

def cci(df, period=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (tp - ma) / (0.015 * md)

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_chart_patterns(df):
    patterns = []
    if len(df) < 10:  # 최소 데이터 길이 증가
        return patterns

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    atr_vals = df['ATR'].values if 'ATR' in df.columns else np.zeros(len(df))

    # W 패턴 (거래량 + 변동성 필터 강화)
    if len(df) >= 7:
        # 5일 전 저점, 현재 저점
        first_low = lows[-7]
        second_low = lows[-1]
        avg_price = (first_low + second_low) / 2
        
        # 조건: 가격 차이 1% 이내, 거래량 증가, 변동성 증가
        if (abs(first_low - second_low) < 0.01 * avg_price and
            volumes[-1] > np.mean(volumes[-7:-2]) * 1.3 and
            atr_vals[-1] > np.mean(atr_vals[-7:-2]) * 0.8):
            patterns.append({
                'name': 'W 패턴',
                'confidence': 85,  # 신뢰도 상향 조정
                'timeframe': '5~15분',
                'movement': '상승',
                'description': "W 패턴 (확률 85%)\n🚀 5~15분 후 상승 확률 ↑"
            })

    # 역 헤드앤숄더 (조건 보완)
    if len(df) >= 9:
        head_low = lows[-5]  # 머리 부분
        left_shoulder = lows[-8]  # 왼쪽 어깨
        right_shoulder = lows[-2]  # 오른쪽 어깨
        
        # 목선 계산 (왼쪽 고점, 오른쪽 고점 평균)
        neckline = (highs[-9] + highs[-3]) / 2
        
        # 조건: 머리가 가장 낮고, 어깨 높이 유사, 목선 돌파, 거래량 증가
        if (left_shoulder > head_low < right_shoulder and
            abs(left_shoulder - right_shoulder) < 0.02 * head_low and
            closes[-1] > neckline and
            volumes[-5] > np.mean(volumes[-9:]) * 1.7):  # 머리 부분에서 거래량 증가
            patterns.append({
                'name': '역 헤드앤숄더',
                'confidence': 90,  # 신뢰도 상향 조정
                'timeframe': '10~30분',
                'movement': '상승',
                'description': "역 헤드앤숄더 (확률 90%)\n🚀 10~30분 후 상승 확률 ↑"
            })

    # 상승 깃발 (추세 지속) - 조건 보완
    if len(df) >= 10:
        # 깃대 부분 (처음 4봉)
        pole = df.iloc[-10:-6]
        # 깃발 부분 (마지막 6봉)
        flag = df.iloc[-6:]
        
        # 깃대 조건: 강한 상승, 높은 거래량
        pole_range = pole['high'] - pole['low']
        pole_avg_range = pole_range.mean()
        
        # 깃발 조건: 수렴, 거래량 감소
        flag_range = flag['high'] - flag['low']
        flag_avg_range = flag_range.mean()
        
        # 가격 변동성 조건
        if (pole_avg_range > 2 * flag_avg_range and
            np.all(pole['close'] > pole['open']) and  # 강한 상승봉
            np.all(flag['volume'] < pole['volume'].mean() * 0.8)):  # 거래량 감소
            patterns.append({
                'name': '상승 깃발패턴',
                'confidence': 80,
                'timeframe': '10~20분',
                'movement': '상승',
                'description': "상승 깃발패턴(추세 지속)\n🚀 10~20분 후 추가 상승 가능성 ↑"
            })

    # 삼각 수렴 (상승/하락 브레이크아웃) - 조건 보완
    if len(df) >= 12:
        last12 = df.iloc[-12:]
        upper = last12['high'].values
        lower = last12['low'].values
        
        # 상한선: 고점 하락
        # 하한선: 저점 상승
        if (np.all(np.diff(upper) < 0) and
            np.all(np.diff(lower) > 0) and
            (upper[-1] - lower[-1]) < 0.7 * (upper[0] - lower[0])):  # 폭이 30% 이상 축소
            patterns.append({
                'name': '삼각 수렴',
                'confidence': 80,  # 신뢰도 상향 조정
                'timeframe': '10~30분',
                'movement': '방향성 돌파',
                'description': "삼각 수렴(돌파 예상)\n⚡ 10~30분 내 방향성 돌파 가능성 ↑"
            })
            
            # 돌파 방향 예측 추가
            if closes[-1] > (upper[-1] + lower[-1]) / 2:
                patterns[-1]['movement'] = '상승 돌파'
                patterns[-1]['description'] = "삼각 수렴(상승 돌파 예상)\n⬆️ 10~30분 내 상승 돌파 가능성 ↑"
            else:
                patterns[-1]['movement'] = '하락 돌파'
                patterns[-1]['description'] = "삼각 수렴(하락 돌파 예상)\n⬇️ 10~30분 내 하락 돌파 가능성 ↑"

    # 추가 패턴: 상승 쐐기 (반전 패턴)
    if len(df) >= 10:
        last10 = df.iloc[-10:]
        upper = last10['high'].values
        lower = last10['low'].values
        
        # 상한선: 고점 상승
        # 하한선: 저점 상승 (더 가파름)
        if (np.all(np.diff(upper) > 0) and
            np.all(np.diff(lower) > 0) and
            np.mean(np.diff(lower)) > 1.5 * np.mean(np.diff(upper))):  # 하한선이 더 가파르게 상승
            patterns.append({
                'name': '상승 쐐기',
                'confidence': 75,
                'timeframe': '15~30분',
                'movement': '하락 반전',
                'description': "상승 쐐기(하락 반전 예측)\n🔻 15~30분 후 하락 반전 가능성 ↑"
            })

    return patterns

def calculate_signal_score(df, latest):
    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []
    latest_rsi = latest.get('RSI', 0) if not df.empty else 0
    
    if df.empty or len(df) < 3:  # 최소 데이터 길이 증가
        return buy_score, sell_score, buy_reasons, sell_reasons, latest_rsi
    
    try:
        required_keys = ['HMA3', 'HMA', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
        if not all(key in latest for key in required_keys):
            raise ValueError("필수 기술적 지표 데이터 누락")

        # 매수 조건
        if latest['HMA3'] > latest['HMA']:
            buy_score += 3
            buy_reasons.append("HMA3 > HMA (3점)")
        
        # RSI 조건 개선
        if 30 < latest['RSI'] < 70:  # 과매수/과매도 구간 제외
            if latest['RSI'] > 45 and latest['RSI'] > df['RSI'].iloc[-2]:
                buy_score += 2
                buy_reasons.append(f"RSI({latest['RSI']:.1f}) > 45 & 상승 (2점)")
        
        # MACD 히스토그램 개선
        if latest['MACD_hist'] > 0 and latest['MACD_hist'] > df.iloc[-2]['MACD_hist']:
            buy_score += 2  # 점수 증가
            buy_reasons.append("MACD 히스토그램 > 0 & 상승 (2점)")
        
        # 거래량 조건 강화
        if len(df) > 2 and latest['volume'] > df.iloc[-2]['volume'] * 1.2:
            buy_score += 2
            buy_reasons.append("거래량 20% 이상 증가 (2점)")
        
        # 볼린저 밴드 조건
        if latest['close'] < latest['BB_lower'] * 1.01:  # 하한선 근접
            buy_score += 2
            buy_reasons.append("가격 < BB 하한선 근접 (2점)")

        # 매도 조건
        if latest['HMA3'] < latest['HMA']:
            sell_score += 3
            sell_reasons.append("HMA3 < HMA (3점)")
        
        # RSI 조건 개선
        if latest['RSI'] > 70:
            sell_score += 3  # 점수 증가
            sell_reasons.append(f"RSI({latest['RSI']:.1f}) > 70 (3점)")
        
        # MACD 히스토그램 개선
        if latest['MACD_hist'] < 0 and latest['MACD_hist'] < df.iloc[-2]['MACD_hist']:
            sell_score += 2  # 점수 증가
            sell_reasons.append("MACD 히스토그램 < 0 & 하락 (2점)")
        
        # 거래량 조건 강화
        if len(df) > 2 and latest['volume'] < df.iloc[-2]['volume'] * 0.8:
            sell_score += 2
            sell_reasons.append("거래량 20% 이상 감소 (2점)")
        
        # 볼린저 밴드 조건
        if latest['close'] > latest['BB_upper'] * 0.99:  # 상한선 근접
            sell_score += 2
            sell_reasons.append("가격 > BB 상한선 근접 (2점)")
            
    except Exception as e:
        st.error(f"신호 계산 오류: {str(e)}")
        buy_reasons = [f"오류 발생: {str(e)}"]
        sell_reasons = [f"오류 발생: {str(e)}"]
    
    return buy_score, sell_score, buy_reasons, sell_reasons, latest_rsi

# 패턴 알림 처리 함수 개선
def process_pattern_alerts(coin, pattern, alert_price):
    now = datetime.now()
    # 패턴 이력에 추가
    pattern_record = {
        'coin': coin,
        'pattern': pattern['description'],
        'name': pattern['name'],
        'confidence': pattern['confidence'],
        'alert_time': now,
        'alert_price': alert_price,
        'current_price': alert_price,
        'completed': False,
        'end_price': None,
        'movement': pattern['movement']
    }
    
    # 중복 패턴 체크
    duplicate = False
    for p in st.session_state.pattern_history:
        if (p['coin'] == coin and 
            p['name'] == pattern['name'] and 
            (now - p['alert_time']).total_seconds() < 600):  # 10분 내 동일 패턴
            duplicate = True
            break
    
    if not duplicate:
        st.session_state.pattern_history.append(pattern_record)
        # 알림 메시지 추가
        alert_msg = f"🔔 [{coin}] 패턴 감지: {pattern['description']} (가격: {alert_price:,.1f})"
        st.session_state.alerts.append(alert_msg)

# ---------------------- 초기 설정 ----------------------
st.set_page_config(layout="wide")
# 형광 녹색 스타일(검정배경+녹색글씨) 적용
DOS_GREEN = "#39FF14"
DOS_BG = "#000000"
st.markdown(
    f"""
    <style>
    body, .stApp {{
        background-color: {DOS_BG} !important;
    }}
    .dos-green, .stMarkdown, .stDataFrame, .stTable, .stText, .stSubheader, .stHeader, .stTitle, .stMetric, .stSidebar, .stSidebarContent, .stSidebar .stTextInput, .stSidebar .stNumberInput, .stSidebar .stCheckbox, .stSidebar .stSelectbox, .stSidebar .stMetric {{
        color: {DOS_GREEN} !important;
        background-color: {DOS_BG} !important;
    }}
    .stMetric label, .stMetric div, .stMetric span {{
        color: {DOS_GREEN} !important;
    }}
    .pattern-alert {{
        border: 1px solid {DOS_GREEN};
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }}
    .completed-pattern {{
        border: 1px solid #FF2222;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }}
    .alert-box {{
        padding: 10px; 
        background: #1a1a1a; 
        border-left: 4px solid {DOS_GREEN};
        border-radius: 4px;
        margin: 10px 0;
    }}
    .buy-signal {{
        color: #00FF00 !important;
        font-weight: bold;
    }}
    .sell-signal {{
        color: #FF2222 !important;
        font-weight: bold;
    }}
    .chart-title {{
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
        color: {DOS_GREEN};
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# 메인 타이틀/서브타이틀 형광녹색, 폰트 두단계 작게(h3/h5)
st.markdown(f"<h3 style='color:{DOS_GREEN};background:{DOS_BG};font-family:Consolas,monospace;'>🔥 If you're not desperate, don't even think about it!</h3>", unsafe_allow_html=True)
st.markdown(f"<h5 style='color:{DOS_GREEN};background:{DOS_BG};font-family:Consolas,monospace;'>🔥 Live Cryptocurrency Analytics Dashboard</h5>", unsafe_allow_html=True)
st_autorefresh(interval=5000, key="auto_refresh")

# ---------------------- 전역 변수 ----------------------
default_holdings = {
    'KRW-STX': 13702.73,
    'KRW-HBAR': 62216.22494886,
    'KRW-DOGE': 61194.37067502,
}
markets = list(default_holdings.keys())
timeframes = {1: '1분', 3: '3분', 5: '5분', 15: '15분', 60: '60분', 240: '240분', 360: '360분'}
TOTAL_INVESTMENT = 58500000
MAIN_COIN = 'KRW-STX'  # 메인 홀딩 코인 지정

# ---------------------- 데이터 함수 (성능 개선) ----------------------
@st.cache_data(ttl=10, show_spinner=False)
def get_current_prices():
    try:
        res = requests.get(f"https://api.upbit.com/v1/ticker?markets={','.join(markets)}", timeout=5)
        if res.status_code == 200:
            return {x['market']: x for x in res.json()}
        else:
            st.error(f"가격 조회 실패: {res.status_code}")
    except Exception as e:
        st.error(f"가격 조회 오류: {str(e)}")
    
    # 실패 시 캐시된 데이터 반환
    if 'cached_prices' in st.session_state:
        return st.session_state.cached_prices
    return {market: {'trade_price': 0, 'signed_change_rate': 0} for market in markets}

@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv(market, timeframe, count=300):
    try:
        url = f"https://api.upbit.com/v1/candles/minutes/{timeframe}"
        params = {'market': market, 'count': count}
        res = requests.get(url, params=params, timeout=10)
        
        if res.status_code != 200:
            st.error(f"{market} 데이터 조회 실패: {res.status_code}")
            return pd.DataFrame()
            
        data = res.json()
        if not data:
            st.warning(f"{market} 데이터 없음")
            return pd.DataFrame()
            
        df = pd.DataFrame(data)[::-1]
        df = df[['candle_date_time_kst','opening_price','high_price','low_price','trade_price','candle_acc_trade_volume']]
        df.columns = ['datetime','open','high','low','close','volume']
        df['datetime'] = pd.to_datetime(df['datetime'])

        # 기술적 지표 계산 (데이터 충분할 때만)
        if len(df) > 20:
            df['HL2'] = (df['high'] + df['low']) / 2
            df['HMA'] = hma(df['HL2'])
            df['HMA3'] = hma3(df['HL2'])
            df['Signal'] = np.where(df['HMA3'] > df['HMA'], '매수', '매도')
            df['RSI'] = rsi(df['close'])
            df['MACD_line'], df['Signal_line'], df['MACD_hist'] = macd(df['close'])
            df['BB_ma'], df['BB_upper'], df['BB_lower'] = bollinger_bands(df['close'])
            df['CCI'] = cci(df)
            df['ATR'] = atr(df)
        return df
    except Exception as e:
        st.error(f"{market} 데이터 처리 오류: {str(e)}")
        return pd.DataFrame()

# ---------------------- 사이드바 설정 ----------------------
with st.sidebar:
    st.markdown(f"<div style='color:{DOS_GREEN};font-family:Consolas,monospace;'>", unsafe_allow_html=True)
    st.header("⚙️ 제어 패널")
    selected_tf = st.selectbox('차트 주기', list(timeframes.keys()), format_func=lambda x: timeframes[x])
    
    st.subheader("💰 투자 현황")
    prices = get_current_prices()
    st.session_state.cached_prices = prices  # 캐시 저장
    
    stx_holding = default_holdings['KRW-STX']
    stx_price = prices.get('KRW-STX', {}).get('trade_price', 0)
    current_value = stx_holding * stx_price
    profit = current_value - TOTAL_INVESTMENT
    profit_percent = (profit / TOTAL_INVESTMENT) * 100 if TOTAL_INVESTMENT else 0
    profit_emoji = "🔻" if profit < 0 else "🟢"

    # 자산가치/등락률: -파랑, +빨강
    profit_color = "#00BFFF" if profit < 0 else "#FF2222"
    st.markdown(f"<div style='color:{DOS_GREEN};'>총 투자금액</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{DOS_GREEN};font-size:22px;'>{TOTAL_INVESTMENT:,.0f} 원</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{DOS_GREEN};'>STX 자산가치</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{profit_color};font-size:22px;'>{current_value:,.0f} 원</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{profit_color};font-size:18px;'>{profit_emoji} {profit:+,.0f} 원 ({profit_percent:+.2f}%)</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{DOS_GREEN};'>STX 보유량</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{DOS_GREEN};font-size:22px;'>{stx_holding:,.2f} EA</div>", unsafe_allow_html=True)
    
    st.subheader("🔔 텔레그램 알림")
    telegram_enabled = st.checkbox("텔레그램 알림 활성화")
    table_alert_interval = st.number_input("테이블 알림 주기(분)", min_value=1, value=10)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- 메인 화면 초기화 ----------------------
def init_session_state():
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'last_alert_time' not in st.session_state:
        st.session_state.last_alert_time = {}
    if 'last_table_alert_time' not in st.session_state:
        st.session_state.last_table_alert_time = datetime.min
    if 'pattern_history' not in st.session_state:
        st.session_state.pattern_history = []
    if 'detected_patterns' not in st.session_state:
        st.session_state.detected_patterns = {market: [] for market in markets}
    if 'cached_prices' not in st.session_state:
        st.session_state.cached_prices = {}

init_session_state()

# ---------------------- 코인 비교 테이블 (성능 개선) ----------------------
def generate_coin_table():
    base_market = 'KRW-STX'
    base_qty = default_holdings[base_market]
    base_price_data = prices.get(base_market, {'trade_price': 0, 'signed_change_rate': 0})
    base_price = base_price_data['trade_price']
    base_krw = base_qty * base_price * 0.9995 if base_price else 0

    compare_data = []
    for market in markets:
        coin = market.split('-')[1]
        price_data = prices.get(market, {'trade_price': 0, 'signed_change_rate': 0})
        price = price_data['trade_price']
        change_rate = price_data.get('signed_change_rate', 0) * 100
        qty = default_holdings[market]
        value = price * qty
        
        # 기술적 지표는 이미 계산된 것 사용 (성능 개선)
        df = fetch_ohlcv(market, selected_tf)
        buy_score, sell_score = 0, 0
        latest_rsi = 0

        if not df.empty and len(df) >= 3:
            try:
                latest = df.iloc[-1]
                if 'RSI' in df.columns:
                    latest_rsi = latest['RSI']
                if 'HMA3' in df.columns and 'HMA' in df.columns:
                    buy_score, sell_score, _, _, _ = calculate_signal_score(df, latest)
            except Exception as e:
                st.error(f"{coin} 신호 계산 오류: {str(e)}")
        
        if market != base_market and price > 0:
            replace_qty = (base_krw * 0.9995) / price
            diff_qty = replace_qty - qty
            replace_value = replace_qty * price * 0.9995
        else:
            replace_qty = "-"
            diff_qty = "-"
            replace_value = "-"
        
        change_color = "red" if change_rate < 0 else "green"
        change_emoji = "🔻" if change_rate < 0 else "🟢"
        buy_color = "green" if buy_score >= 7 else "gray"
        sell_color = "red" if sell_score >= 7 else "gray"
        
        compare_data.append({
            '코인명': coin,
            '시세': f"{price:,.1f} 원" if price > 0 else "-",
            'RSI': f"{latest_rsi:.1f}" if latest_rsi > 0 else "-",
            '매수신호': f"<span style='color:{buy_color}'>매수({buy_score}/10)</span>",
            '매도신호': f"<span style='color:{sell_color}'>매도({sell_score}/10)</span>",
            '등락률': f"<span style='color:{change_color}'>{change_emoji} {change_rate:+.2f}%</span>",
            '보유수량': qty,
            '대체가능수량': replace_qty if market != base_market else "-",
            '차이수량': diff_qty if market != base_market else "-",
            '평가금액': value,
            '대체평가액': replace_value if market != base_market else "-"
        })

    return pd.DataFrame(compare_data)

def format_number(x):
    if isinstance(x, (int, float)):
        if x == "-": return x
        return f"{x:,.2f}" if abs(x) < 10000 else f"{x:,.0f}"
    return x

# ---------------------- 탭 구성 ----------------------
tab1, tab2, tab3 = st.tabs(["📊 코인 비교 테이블", "📈 코인 분석", "🔮 코인 예측"])

with tab1:

    styled = (
        df_compare.style.format({
            '보유수량': format_number,
            '대체가능수량': format_number,
            '차이수량': lambda x: f"{x:+.0f}" if isinstance(x, (int, float)) else x,
            '평가금액': format_number,
            '대체평가액': format_number,
            'RSI': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
        })
        .map(diff_qty_color, subset=['차이수량'])
        .map(change_rate_color, subset=['등락률'])
        .map(lambda _: 'text-align: center')
    )

     st.subheader("📊 코인 비교 테이블 (RSI 포함)")
    df_compare = generate_coin_table()
    
    # 차이수량 컬럼: -파랑, +빨강, 0은 녹색
    def diff_qty_color(val):
        try:
            v = float(val)
            if v > 0:
                return "color:#FF2222"
            elif v < 0:
                return "color:#00BFFF"
            else:
                return f"color:{DOS_GREEN}"
        except Exception:
            return f"color:{DOS_GREEN}"

    # 등락률 컬럼: -파랑, +빨강
    def change_rate_color(val):
        try:
            num = float(re.search(r'[-+]?\d*\.\d+|\d+', val).group())
            if num < 0:
                return "color:#00BFFF"
            elif num > 0:
                return "color:#FF2222"
            else:
                return f"color:{DOS_GREEN}"
        except Exception:
            return f"color:{DOS_GREEN}"
   
    st.markdown(styled.to_html(escape=False, index=False), unsafe_allow_html=True)

with tab2:

# ---------------------- 테이블 알림 생성 (개선) ----------------------
now = datetime.now()
tf_str = timeframes[selected_tf]
if (now - st.session_state.last_table_alert_time) > timedelta(minutes=table_alert_interval):
    alert_msg = "📊 분석현황(by MIWOONI)\n\n"
    for _, row in df_compare.iterrows():
        alert_msg += (
            f"[{row['코인명']}]\n"
            f"시세: {row['시세']}\n"
            f"RSI({tf_str}): {row['RSI']}\n"
            f"매수신호: {row['매수신호'].split('>')[1].split('<')[0]}\n"
            f"매도신호: {row['매도신호'].split('>')[1].split('<')[0]}\n"
            f"보유량: {row['보유수량']:,.2f}\n"
            f"평가금액: {row['평가금액']:,.0f}원\n"
        )
        if row['코인명'].upper() in ['HBAR', 'DOGE']:
            alert_msg += (
                f"대체 가능 수량: {row['대체가능수량']:,.2f}\n"
                f"차이 수량: {row['차이수량']:+,.2f}\n"
            )
        alert_msg += "\n"
    
    st.session_state.last_table_alert_time = now
    if telegram_enabled:
        send_telegram_alert(alert_msg.strip())
    st.session_state.alerts.append(alert_msg)

    # RSI 비교 차트
    st.subheader("📈 RSI 비교 차트")
    fig_rsi = make_subplots(rows=1, cols=1)
    rsi_time_window = timedelta(days=3)  # 기간 단축 (성능 개선)
    now_time = datetime.now()
    
    for market in markets:
        coin = market.split('-')[1]
        df = fetch_ohlcv(market, selected_tf)
        if not df.empty and 'RSI' in df.columns:
            df_recent = df[df['datetime'] >= now_time - rsi_time_window]
            if not df_recent.empty:
                fig_rsi.add_trace(go.Scatter(
                    x=df_recent['datetime'],
                    y=df_recent['RSI'],
                    name=f'{coin} RSI',
                    line=dict(width=2)
                ))
    
    if len(fig_rsi.data) > 0:
        fig_rsi.update_layout(
            height=400,
            title=f"RSI 비교 ({timeframes[selected_tf]} 차트)",
            yaxis_title="RSI",
            hovermode="x unified"
        )
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        st.plotly_chart(fig_rsi, use_container_width=True)
    else:
        st.warning("RSI 데이터가 없습니다")

# ---------------------- 개별 코인 분석 (가독성 개선) ----------------------
for market in markets:
    coin = market.split('-')[1]
    with st.container():
        st.markdown(f"<div class='chart-title'>{coin} 차트 분석 ({timeframes[selected_tf]})</div>", unsafe_allow_html=True)
        
        df = fetch_ohlcv(market, selected_tf)
        if df.empty:
            st.warning(f"{coin} 차트 데이터 불러오기 실패")
            continue

        latest = df.iloc[-1] if len(df) > 0 else {}
        current_price = latest.get('close', 0)
        prev_close = df.iloc[-2]['close'] if len(df) > 1 else current_price
        delta = current_price - prev_close
        delta_percent = (delta / prev_close) * 100 if prev_close else 0

        # 신호 점수 계산
        buy_score, sell_score, buy_reasons, sell_reasons, _ = calculate_signal_score(df, latest)
        
        # 신호 상태 표시
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='buy-signal'>매수 신호: {buy_score}/10</div>", unsafe_allow_html=True)
            for reason in buy_reasons:
                st.markdown(f"<div>✓ {reason}</div>", unsafe_allow_html=True)
                
        with col2:
            st.markdown(f"<div class='sell-signal'>매도 신호: {sell_score}/10</div>", unsafe_allow_html=True)
            for reason in sell_reasons:
                st.markdown(f"<div>✓ {reason}</div>", unsafe_allow_html=True)
        
        # 차트 그리기
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
            row_heights=[0.6, 0.2, 0.2],
            specs=[[{"secondary_y": True}], [{}], [{}]]
        )
        
        # 캔들스틱 차트
        fig.add_trace(go.Candlestick(
            x=df['datetime'], 
            open=df['open'], 
            high=df['high'],
            low=df['low'], 
            close=df['close'], 
            name="Price"
        ), row=1, col=1)
        
        # 현재가 표시
        if current_price > 0:
            fig.add_hline(
                y=current_price, 
                line_dash="solid", 
                line_color="cyan", 
                row=1, col=1,
                annotation_text=f"현재가: {current_price:,.1f}", 
                annotation_position="top left",
                annotation_font_color="cyan",
                annotation_font_size=12
            )
        
        # 기술적 지표 (데이터 있을 때만)
        if 'HMA' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['datetime'], 
                y=df['HMA'], 
                name='HMA', 
                line=dict(color='blue')
            ), row=1, col=1)
            
        if 'HMA3' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['datetime'], 
                y=df['HMA3'], 
                name='HMA3', 
                line=dict(color='orange')
            ), row=1, col=1)
            
        if 'BB_upper' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['datetime'], 
                y=df['BB_upper'], 
                name='BB Upper', 
                line=dict(color='gray', dash='dot')
            ), row=1, col=1)
            
        if 'BB_lower' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['datetime'], 
                y=df['BB_lower'], 
                name='BB Lower', 
                line=dict(color='gray', dash='dot')
            ), row=1, col=1)
        
        # 거래량
        fig.add_trace(go.Bar(
            x=df['datetime'], 
            y=df['volume'], 
            name='Volume',
            marker_color=np.where(df['close'] > df['open'], 'green', 'red')
        ), row=1, col=1, secondary_y=True)
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['datetime'], 
                y=df['RSI'], 
                name='RSI', 
                line=dict(color='purple')
            ), row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
        
        # MACD
        if 'MACD_hist' in df.columns:
            fig.add_trace(go.Bar(
                x=df['datetime'], 
                y=df['MACD_hist'], 
                name='Histogram',
                marker_color=np.where(df['MACD_hist'] > 0, 'green', 'red')
            ), row=3, col=1)
            
        if 'MACD_line' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['datetime'], 
                y=df['MACD_line'], 
                name='MACD', 
                line=dict(color='blue')
            ), row=3, col=1)
            
        if 'Signal_line' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['datetime'], 
                y=df['Signal_line'], 
                name='Signal', 
                line=dict(color='orange')
            ), row=3, col=1)

        fig.update_layout(
            height=700,  # 높이 조정
            title=f"{coin} 차트 ({selected_tf})",
            xaxis_rangeslider_visible=False,
            margin=dict(t=40, b=40),
            showlegend=True,
            template="plotly_dark"  # 어두운 테마 적용
        )
        
        # X축 범위 설정 (최근 50개 데이터만 표시)
        if len(df) > 50:
            fig.update_xaxes(range=[df['datetime'].iloc[-50], df['datetime'].iloc[-1]])

        st.plotly_chart(fig, use_container_width=True)
        
        # 패턴 감지 및 저장 (모든 코인에 대해)
        detected_patterns = detect_chart_patterns(df)
        st.session_state.detected_patterns[market] = detected_patterns
        
        # 메인 코인(STX)에 대해서만 패턴 알림
        if market == MAIN_COIN and detected_patterns:
            pattern_alerts = []
            for pattern in detected_patterns:
                timeframe_str = timeframes[selected_tf]
                alert_msg = f"🚨🚨 [{coin} {timeframe_str}차트] {current_price:,.1f}원 // {pattern['description']} ({now.strftime('%H:%M:%S')})"
                
                # 중복 알림 방지
                last_pattern_alert_key = f"{coin}_{pattern['name']}_pattern"
                last_alert_time = st.session_state.last_alert_time.get(last_pattern_alert_key, datetime.min)
                
                if (now - last_alert_time) > timedelta(minutes=10):  # 10분 간격으로만 알림
                    if telegram_enabled:
                        send_telegram_alert(alert_msg)
                    st.session_state.last_alert_time[last_pattern_alert_key] = now
                    pattern_alerts.append(alert_msg)
            
            st.session_state.alerts.extend(pattern_alerts)

# ---------------------- 패턴 알림 처리 로직 (메인 코인만) ----------------------
if MAIN_COIN in st.session_state.detected_patterns:
    coin = MAIN_COIN.split('-')[1]
    detected_patterns = st.session_state.detected_patterns[MAIN_COIN]
    current_price = prices.get(MAIN_COIN, {}).get('trade_price', 0)
    
    for pattern in detected_patterns:
        process_pattern_alerts(coin, pattern, current_price)

# ---------------------- 코인 예측 탭 (가독성 개선) ----------------------
with tab3:
    st.subheader("🔮 코인 예측 패턴 분석")
    
    prediction_data = []
    for market in markets:
        coin = market.split('-')[1]
        detected_patterns = st.session_state.detected_patterns.get(market, [])
        
        for pattern in detected_patterns:
            prediction_data.append({
                '코인': coin,
                '패턴명': pattern['name'],
                '확률(%)': pattern['confidence'],
                '예상 시간대': pattern['timeframe'],
                '예상 방향': pattern['movement'],
                '신호 강도': '🔴 강함' if pattern['confidence'] >= 80 else '🟡 보통'
            })
    
    if prediction_data:
        df_predictions = pd.DataFrame(prediction_data)
        
        # 확률에 따라 내림차순 정렬
        df_predictions = df_predictions.sort_values(by='확률(%)', ascending=False)
        
        # 신호 강도에 따라 색상 지정
        def color_strength(val):
            if '강함' in val:
                return "background-color: #330000; color: #FF5555; font-weight: bold;"
            return "background-color: #333300; color: #FFFF55;"
        
        # 방향에 따라 색상 지정
        def color_direction(val):
            if '상승' in val or '돌파' in val:
                return "color: #00FF00;"
            elif '하락' in val or '반전' in val:
                return "color: #FF5555;"
            return ""
        
        styled = (
            df_predictions.style
            .applymap(color_strength, subset=['신호 강도'])
            .applymap(color_direction, subset=['예상 방향'])
            .format({'확률(%)': '{:.0f}%'})
            .set_properties(**{'text-align': 'center'})
        )
        st.markdown(styled.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("현재 감지된 패턴이 없습니다")

# ---------------------- 실시간 패턴 업데이트 ----------------------
if 'minute_counter' not in st.session_state:
    st.session_state.minute_counter = 0

st.session_state.minute_counter += 1
if st.session_state.minute_counter % 12 == 0:  # 5초 * 12 = 60초
    current_prices = get_current_prices()
    active_patterns = [p for p in st.session_state.pattern_history if not p.get('completed')]
    
    for pattern in active_patterns:
        market = f"KRW-{pattern['coin']}"
        if market in current_prices:
            pattern['current_price'] = current_prices[market]['trade_price']
            
        # 15분이 지난 패턴을 완료 처리
        elapsed_minutes = (datetime.now() - pattern['alert_time']).total_seconds() / 60
        if elapsed_minutes >= 15:
            pattern['completed'] = True
            pattern['end_price'] = pattern['current_price']

# ---------------------- 사이드바 패턴 정보 표시 (개선) ----------------------
with st.sidebar:
    st.subheader("🔮 패턴 예측 분석")
    
    # 활성 패턴
    st.markdown(f"<div style='color:{DOS_GREEN};font-weight:bold;'>🔔 활성 패턴 알림</div>", unsafe_allow_html=True)
    active_patterns = [p for p in st.session_state.pattern_history if not p.get('completed')]
    
    if not active_patterns:
        st.markdown(f"<div style='color:{DOS_GREEN};'>- 활성 알림 없음</div>", unsafe_allow_html=True)
    else:
        for pattern in active_patterns:
            elapsed_min = (datetime.now() - pattern['alert_time']).seconds // 60
            price_diff = pattern['current_price'] - pattern['alert_price']
            price_diff_percent = (price_diff / pattern['alert_price']) * 100
            
            # 가격 변동 색상
            diff_color = "#FF2222" if price_diff < 0 else "#00BFFF"
            diff_emoji = "🔻" if price_diff < 0 else "🟢"
            
            st.markdown(
                f"<div class='pattern-alert'>"
                f"<div style='color:{DOS_GREEN};'>"
                f"▫️ <b>{pattern['coin']}</b> ({pattern['name']})<br>"
                f"- {pattern['pattern'].split('(')[0]}<br>"
                f"- 예측시간: {pattern['alert_time'].strftime('%H:%M:%S')}<br>"
                f"- 예측가격: {pattern['alert_price']:,.1f}원<br>"
                f"- 현재가격: <span style='color:{diff_color};'>{pattern['current_price']:,.1f}원</span><br>"
                f"- 변동: <span style='color:{diff_color};'>{diff_emoji} {price_diff:+,.1f}원 ({price_diff_percent:+.2f}%)</span><br>"
                f"- 경과시간: {elapsed_min}분"
                f"</div></div>",
                unsafe_allow_html=True
            )
    
    # 완료된 패턴
    st.markdown(f"<div style='color:{DOS_GREEN};font-weight:bold;margin-top:20px;'>📊 최근 패턴 결과</div>", unsafe_allow_html=True)
    completed_patterns = [p for p in st.session_state.pattern_history if p.get('completed')][-3:]
    
    if not completed_patterns:
        st.markdown(f"<div style='color:{DOS_GREEN};'>- 완료된 알림 없음</div>", unsafe_allow_html=True)
    else:
        for pattern in completed_patterns:
            change_percent = (pattern['end_price'] - pattern['alert_price']) / pattern['alert_price'] * 100
            change_color = "#FF2222" if change_percent < 0 else "#00BFFF"
            result_emoji = "❌" if change_percent < 0 else "✅"
            
            st.markdown(
                f"<div class='completed-pattern'>"
                f"<div style='color:{DOS_GREEN};'>"
                f"▫️ <b>{pattern['coin']}</b> ({pattern['name']}) {result_emoji}<br>"
                f"- 예측: {pattern['movement']}<br>"
                f"- 예측가: {pattern['alert_price']:,.1f}원<br>"
                f"- 종료가: {pattern['end_price']:,.1f}원<br>"
                f"- 결과: <span style='color:{change_color};'>{change_percent:+.2f}%</span><br>"
                f"- 경과시간: 15분"
                f"</div></div>",
                unsafe_allow_html=True
            )

# ---------------------- 실시간 알림 출력 (디자인 개선) ----------------------
st.subheader("🔔 분석알림 (RSI 및 패턴 포함)")
if st.session_state.alerts:
    for alert in reversed(st.session_state.alerts[-10:]):
        # 패턴 알림 강조
        if "패턴 감지" in alert:
            st.markdown(f"<div class='alert-box' style='border-left: 4px solid #FF9900;'>🚨 {alert}</div>", unsafe_allow_html=True)
        # 매수/매도 신호 강조
        elif "매수" in alert or "매도" in alert:
            border_color = "#00FF00" if "매수" in alert else "#FF2222"
            st.markdown(f"<div class='alert-box' style='border-left: 4px solid {border_color};'>📢 {alert}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='alert-box'>ℹ️ {alert}</div>", unsafe_allow_html=True)
else:
    st.info("최근 알림이 없습니다")

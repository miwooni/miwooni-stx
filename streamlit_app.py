import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
import os
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input

# ✅ 첫 번째 Streamlit 명령어
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------- AI 예측 모델 ----------------------
def create_prediction_model():
    model = Sequential([
        Input(shape=(60, 1)),
        LSTM(50, activation='relu', return_sequences=False),
        Dense(25, activation='relu'),
        Dense(5)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def apply_prediction_correction(pred_prices, actual_prices):
    if len(actual_prices) < 5 or len(pred_prices) < 5:
        return pred_prices
    errors = []
    for i in range(1, min(6, len(actual_prices))):
        if len(pred_prices) >= i:
            errors.append(actual_prices[-i] - pred_prices[-i])
    if not errors:
        return pred_prices
    avg_error = sum(errors) / len(errors)
    return [p + avg_error * 0.7 for p in pred_prices]

# ---------------------- 공통 함수 정의 ----------------------
def send_telegram_alert(message: str):
    bot_token = st.session_state.telegram_bot_token
    chat_id = st.session_state.telegram_chat_id
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        res = requests.post(url, data=payload, timeout=5)
        if res.status_code != 200:
            st.error(f"Telegram 전송 실패: {res.status_code} {res.text}")
        else:
            st.session_state.alerts.append(f"텔레그램 전송: {message}")
    except Exception as e:
        st.error(f"Telegram 알림 전송 실패: {str(e)}")

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

# 개선된 신호 점수 계산 함수
def calculate_signal_score(df, latest, pred_prices=None):
    buy_score, sell_score, risk_score = 0, 0, 0
    buy_reasons, sell_reasons, risk_reasons = [], [], []
    latest_rsi = latest.get('RSI', 0) if not df.empty else 0

    WEIGHTS = {
        'hma_cross': 1.5,
        'rsi': 1.0,
        'macd': 1.2,
        'volume': 1.0,
        'bb_breakout': 1.3,
        'pattern': 2.0,
        'ai_predict': 2.0,
        'risk': 1.0
    }

    try:
        required_keys = ['HMA3', 'HMA', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
        if not all(key in latest for key in required_keys):
            raise ValueError("필수 기술적 지표 데이터 누락")

        # HMA 교차
        if latest['HMA3'] > latest['HMA']:
            buy_score += WEIGHTS['hma_cross'] * 2
            buy_reasons.append("HMA3 > HMA (가중 2점)")
        if latest['HMA3'] < latest['HMA']:
            sell_score += WEIGHTS['hma_cross'] * 2
            sell_reasons.append("HMA3 < HMA (가중 2점)")

        # RSI
        if latest['RSI'] > 30:
            buy_score += WEIGHTS['rsi'] * 1
            buy_reasons.append("RSI > 30")
        if latest['RSI'] > 70:
            sell_score += WEIGHTS['rsi'] * 2
            sell_reasons.append("RSI > 70")

        # MACD 히스토그램
        if latest['MACD_hist'] > 0:
            buy_score += WEIGHTS['macd'] * 1
            buy_reasons.append("MACD > 0")
        if latest['MACD_hist'] < 0:
            sell_score += WEIGHTS['macd'] * 1
            sell_reasons.append("MACD < 0")

        # 거래량
        if len(df) > 1 and latest['volume'] > df.iloc[-2]['volume']:
            buy_score += WEIGHTS['volume'] * 1
            buy_reasons.append("거래량 증가")
        if len(df) > 1 and latest['volume'] < df.iloc[-2]['volume']:
            sell_score += WEIGHTS['volume'] * 1
            sell_reasons.append("거래량 감소")

        # 볼린저 밴드 돌파
        if latest['close'] < latest['BB_lower']:
            buy_score += WEIGHTS['bb_breakout'] * 1
            buy_reasons.append("BB 하단 이탈")
        if latest['close'] > latest['BB_upper']:
            sell_score += WEIGHTS['bb_breakout'] * 1
            sell_reasons.append("BB 상단 돌파")

        # AI 예측 반영
        if pred_prices is not None and len(pred_prices) > 0:
            predicted = pred_prices[-1]
            if predicted > latest['close']:
                buy_score += WEIGHTS['ai_predict'] * 1
                buy_reasons.append("AI 예측 상승")
            elif predicted < latest['close']:
                sell_score += WEIGHTS['ai_predict'] * 1
                sell_reasons.append("AI 예측 하락")

        # 리스크
        if abs(df['close'].pct_change().iloc[-1]) > 0.03:
            risk_score += 1
            risk_reasons.append("단기 급등/급락 경고")
        if latest['volume'] > df['volume'].mean() * 4:
            risk_score += 1
            risk_reasons.append("비정상 거래량")

    except Exception as e:
        buy_reasons.append(f"오류 발생: {str(e)}")
        sell_reasons.append(f"오류 발생: {str(e)}")

    return buy_score, sell_score, risk_score, buy_reasons, sell_reasons, risk_reasons, latest_rsi

# ---------------------- 초기 설정 ----------------------
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
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(f"<h3 style='color:{DOS_GREEN};background:{DOS_BG};font-family:Consolas,monospace;'>🔥 If you're not desperate, don't even think about it!</h3>", unsafe_allow_html=True)
st.markdown(f"<h5 style='color:{DOS_GREEN};background:{DOS_BG};font-family:Consolas,monospace;'>🔥 Live Cryptocurrency Analytics Dashboard</h5>", unsafe_allow_html=True)
st_autorefresh(interval=5000, key="auto_refresh")

# ---------------------- 전역 변수 ----------------------
default_holdings = {
    'KRW-STX': 13702.725635935,
    'KRW-HBAR': 62216.22494886,
    'KRW-DOGE': 61194.37067502,
}
markets = list(default_holdings.keys())
timeframes = {1: '1분', 3: '3분', 5: '5분', 15: '15분', 60: '60분', 240: '240분'}
TOTAL_INVESTMENT = 58500000

# ---------------------- 데이터 함수 ----------------------
@st.cache_data(ttl=10)
def get_current_prices():
    try:
        res = requests.get(f"https://api.upbit.com/v1/ticker?markets={','.join(markets)}")
        return {x['market']:x for x in res.json()}
    except:
        return {market: {'trade_price': 0, 'signed_change_rate': 0} for market in markets}

@st.cache_data(ttl=30)
def fetch_ohlcv(market, timeframe, count=300):
    try:
        url = f"https://api.upbit.com/v1/candles/minutes/{timeframe}"
        params = {'market': market, 'count': count}
        res = requests.get(url, params=params, timeout=10)
        df = pd.DataFrame(res.json())[::-1]
        df = df[['candle_date_time_kst','opening_price','high_price','low_price','trade_price','candle_acc_trade_volume']]
        df.columns = ['datetime','open','high','low','close','volume']
        df['datetime'] = pd.to_datetime(df['datetime'])

        df['HL2'] = (df['high'] + df['low']) / 2
        df['HMA'] = hma(df['HL2'])
        df['HMA3'] = hma3(df['HL2'])
        df['Signal'] = np.where(df['HMA3'] > df['HMA'], '매수', '매도')
        df['RSI'] = rsi(df['close'])
        df['MACD_line'], df['Signal_line'], df['MACD_hist'] = macd(df['close'])
        df['BB_ma'], df['BB_upper'], df['BB_lower'] = bollinger_bands(df['close'])
        return df
    except:
        return pd.DataFrame()

# ---------------------- 세션 상태 초기화 ----------------------
if 'holdings' not in st.session_state:
    st.session_state.holdings = default_holdings.copy()
if 'total_investment' not in st.session_state:
    st.session_state.total_investment = TOTAL_INVESTMENT
if 'telegram_bot_token' not in st.session_state:
    st.session_state.telegram_bot_token = "7545404316:AAHMdayWZjwEZmZwd5JrKXPDn5wUQfivpTw"
if 'telegram_chat_id' not in st.session_state:
    st.session_state.telegram_chat_id = "7890657899"
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# ---------------------- 사이드바 ----------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("📢 알림 내역")
    alert_history = st.session_state.get("alerts", [])
    if alert_history:
        for alert in reversed(alert_history[-10:]):
            st.markdown(
                f"<div style='background:#222;padding:8px 10px;margin:4px 0;border-radius:6px;font-size:14px;color:#39FF14;'>{alert}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("알림 내역이 없습니다.")
    
    st.subheader("⚙️ 설정")
    selected_tf = st.selectbox("차트 주기", list(timeframes.keys()), 
                              format_func=lambda x: timeframes[x], key="selected_tf", index=2)
    telegram_enabled = st.checkbox("✉️ 텔레그램 알림 활성화", value=False)
    st.session_state.telegram_bot_token = st.text_input(
        "텔레그램 봇 토큰",
        value=st.session_state.telegram_bot_token
    )
    st.session_state.telegram_chat_id = st.text_input(
        "텔레그램 채팅 ID",
        value=st.session_state.telegram_chat_id
    )
    
    st.subheader("💰 보유 코인")
    for market in markets:
        coin = market.split('-')[1]
        st.session_state.holdings[market] = st.number_input(
            f"{coin} 보유량",
            value=st.session_state.holdings.get(market, 0.0),
            step=0.0001,
            format="%.4f",
            key=f"holding_{market}"
        )
    st.session_state.total_investment = st.number_input(
        "총 투자금액 (원)",
        value=st.session_state.total_investment,
        step=100000,
        format="%d"
    )

# ---------------------- 메인 코인 비교 테이블 ----------------------
def generate_coin_table(selected_tf):
    signal_scores = {} 
    prices = get_current_prices()
    base_market = 'KRW-STX'
    base_qty = st.session_state.holdings[base_market]
    base_price_data = prices.get(base_market, {'trade_price': 0, 'signed_change_rate': 0})
    base_price = base_price_data['trade_price']
    base_krw = base_qty * base_price * 0.9995

    compare_data = []
    for market in markets:
        coin = market.split('-')[1]
        price_data = prices.get(market, {'trade_price': 0, 'signed_change_rate': 0})
        price = price_data['trade_price']
        change_rate = price_data['signed_change_rate'] * 100
        qty = st.session_state.holdings[market]
        value = price * qty
        
        df = fetch_ohlcv(market, selected_tf)
        buy_score, sell_score = 0, 0
        latest_rsi = 0

        if not df.empty and len(df) >= 2:
            try:
                latest = df.iloc[-1]
                required_cols = ['HMA', 'HMA3', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
                if all(col in df.columns for col in required_cols):
                    buy_score, sell_score, _, _, _, _, latest_rsi = calculate_signal_score(df, latest, pred_prices=None)
                else:
                    st.error(f"{coin} 데이터 컬럼 누락")
            except Exception as e:
                st.error(f"{coin} 신호 계산 오류: {str(e)}")
        else:
            st.warning(f"{coin} 데이터 부족으로 분석 생략")
        
        if market != base_market:
            replace_qty = (base_krw * 0.9995) / price if price else 0
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
            '시세': f"{price:,.1f} 원",
            'RSI': f"{latest_rsi:.1f}",
            '매수신호': f"<span style='color:{buy_color}'>매수({buy_score}/10)</span>",
            '매도신호': f"<span style='color:{sell_color}'>매도({sell_score}/10)</span>",
            '등락률': f"<span style='color:{change_color}'>{change_emoji} {change_rate:+.2f}%</span>",
            '보유수량': qty,
            '대체가능수량': replace_qty if market != base_market else "-",
            '차이수량': diff_qty if market != base_market else "-",
            '평가금액': f"{value:,.0f}",
            '대체평가액': f"{replace_value:,.0f}" if market != base_market else "-"
        })

    return pd.DataFrame(compare_data)

# ---------------------- 이동평균선 교차 알림 ----------------------
def check_ma_cross_alerts():
    if 'ma_cross_alerts' not in st.session_state:
        st.session_state.ma_cross_alerts = {}
    
    for market in markets:
        coin = market.split('-')[1]
        df = fetch_ohlcv(market, selected_tf)
        if len(df) > 20:
            df['SMA5'] = df['close'].rolling(5).mean()
            df['SMA20'] = df['close'].rolling(20).mean()
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            golden_cross = (prev['SMA5'] < prev['SMA20']) and (latest['SMA5'] > latest['SMA20'])
            dead_cross = (prev['SMA5'] > prev['SMA20']) and (latest['SMA5'] < latest['SMA20'])
            now = datetime.now()
            alert_key = f"{market}_cross"
            last_alert = st.session_state.ma_cross_alerts.get(alert_key, datetime.min)
            
            if (golden_cross or dead_cross) and (now - last_alert) > timedelta(minutes=10):
                msg_type = "골든크로스" if golden_cross else "데드크로스"
                message = f"🚨 {coin} {msg_type} 발생! ({now.strftime('%H:%M:%S')})"
                st.session_state.alerts.append(message)
                st.session_state.ma_cross_alerts[alert_key] = now
                if telegram_enabled:
                    send_telegram_alert(message)

# ---------------------- 메인 실행 ----------------------
st.subheader("📊 실시간 코인 분석 테이블")
df_compare = generate_coin_table(selected_tf)

# 알림 체크 실행
check_ma_cross_alerts()

# 테이블 스타일링 함수
def diff_qty_color(val):
    try:
        v = float(val)
        if v > 0: return "color:#FF2222"
        elif v < 0: return "color:#00BFFF"
        else: return f"color:{DOS_GREEN}"
    except: return f"color:{DOS_GREEN}"

def change_rate_color(val):
    try:
        import re
        match = re.search(r'([+\-]?\d+(\.\d+)?)', str(val))
        num = float(match.group(1)) if match else 0
        if '-' in str(val): return "color:#00BFFF"
        elif '+' in str(val): return "color:#FF2222"
        elif num > 0: return "color:#FF2222"
        elif num < 0: return "color:#00BFFF"
        else: return f"color:{DOS_GREEN}"
    except: return f"color:{DOS_GREEN}"

# 테이블 표시
if not df_compare.empty:
    st.markdown("""
    <style>
    .table-wrapper {
        overflow-x: auto;
        margin-top: 20px;
        border: 1px solid #39FF14;
        border-radius: 5px;
        padding: 10px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th {
        background-color: #222;
        color: #39FF14;
        padding: 10px;
        text-align: center;
        border: 1px solid #39FF14;
    }
    td {
        padding: 8px 10px;
        text-align: center;
        border: 1px solid #39FF14;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="table-wrapper">' + df_compare.to_html(escape=False, index=False) + '</div>', unsafe_allow_html=True)
else:
    st.warning("데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.")

# ---------------------- 하단 상태 바 ----------------------
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f"<div style='text-align:right; color:{DOS_GREEN}; margin-top:20px;'>Last Updated: {current_time}</div>", 
            unsafe_allow_html=True)

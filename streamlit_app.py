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
import base64
import os
from tenacity import retry, stop_after_attempt, wait_fixed

# ---------------------- 공통 함수 정의 (성능 개선) ----------------------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def send_telegram_alert(message: str):
    bot_token = "7545404316:AAHMdayWZjwEZmZwd5JrKXPDn5wUQfivpTw"
    chat_id = "7890657899"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    
    try:
        res = requests.post(url, data=payload, timeout=10)
        if res.status_code != 200:
            st.error(f"Telegram 전송 실패: {res.status_code} {res.text}")
        else:
            if 'alerts' in st.session_state:
                st.session_state.alerts.append(f"텔레그램 전송: {message}")
    except Exception as e:
        st.error(f"Telegram 알림 전송 실패: {str(e)}")
        raise

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

# 패턴 감지 알고리즘 정밀화
def detect_chart_patterns(df):
    patterns = []
    if len(df) < 10:
        return patterns

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    atr_vals = df['ATR'].values if 'ATR' in df.columns else np.zeros(len(df))

    # W 패턴 - 조건 강화
    if len(df) >= 7:
        first_low = lows[-7]
        second_low = lows[-1]
        middle_high = highs[-4]  # 중간 고점
        
        avg_price = (first_low + second_low) / 2
        price_diff = abs(first_low - second_low)
        vol_ratio = volumes[-1] / np.mean(volumes[-7:-2])
        atr_ratio = atr_vals[-1] / np.mean(atr_vals[-7:-2])
        
        # 조건: 가격 차이 1% 이내, 거래량 30% 이상 증가, 변동성 증가, 중간 고점이 양쪽 저점보다 높음
        if (price_diff < 0.01 * avg_price and
            vol_ratio > 1.3 and
            atr_ratio > 0.8 and
            middle_high > first_low * 1.01 and
            middle_high > second_low * 1.01):
            patterns.append({
                'name': 'W 패턴',
                'confidence': min(85 + int((vol_ratio - 1.3) * 20), 95),
                'timeframe': '5~15분',
                'movement': '상승',
                'description': "W 패턴 (확률 {}%)\n🚀 5~15분 후 상승 확률 ↑".format(min(85 + int((vol_ratio - 1.3) * 20), 95))
            })

    # 역 헤드앤숄더 - 조건 보완
    if len(df) >= 9:
        head_low = lows[-5]
        left_shoulder = lows[-8]
        right_shoulder = lows[-2]
        neckline = (highs[-9] + highs[-3]) / 2
        
        shoulder_diff = abs(left_shoulder - right_shoulder)
        vol_ratio = volumes[-5] / np.mean(volumes[-9:])
        
        # 조건: 머리가 가장 낮고, 어깨 높이 유사, 목선 돌파, 거래량 70% 이상 증가
        if (left_shoulder > head_low < right_shoulder and
            shoulder_diff < 0.02 * head_low and
            closes[-1] > neckline and
            vol_ratio > 1.7):
            confidence = min(90 + int((vol_ratio - 1.7) * 15), 98)
            patterns.append({
                'name': '역 헤드앤숄더',
                'confidence': confidence,
                'timeframe': '10~30분',
                'movement': '상승',
                'description': f"역 헤드앤숄더 (확률 {confidence}%)\n🚀 10~30분 후 상승 확률 ↑"
            })

    # 상승 깃발 - 조건 보완
    if len(df) >= 10:
        pole = df.iloc[-10:-6]
        flag = df.iloc[-6:]
        
        pole_range = pole['high'] - pole['low']
        pole_avg_range = pole_range.mean()
        flag_range = flag['high'] - flag['low']
        flag_avg_range = flag_range.mean()
        vol_ratio = np.mean(flag['volume']) / np.mean(pole['volume'])
        
        if (pole_avg_range > 2 * flag_avg_range and
            np.all(pole['close'] > pole['open']) and
            vol_ratio < 0.8):
            patterns.append({
                'name': '상승 깃발패턴',
                'confidence': 80,
                'timeframe': '10~20분',
                'movement': '상승',
                'description': "상승 깃발패턴(추세 지속)\n🚀 10~20분 후 추가 상승 가능성 ↑"
            })

    # 삼각 수렴 - 조건 보완
    if len(df) >= 12:
        last12 = df.iloc[-12:]
        upper = last12['high'].values
        lower = last12['low'].values
        
        upper_diff = np.diff(upper)
        lower_diff = np.diff(lower)
        range_ratio = (upper[-1] - lower[-1]) / (upper[0] - lower[0])
        
        if (np.all(upper_diff < 0) and
            np.all(lower_diff > 0) and
            range_ratio < 0.7):
            confidence = 80
            pattern_desc = "삼각 수렴(돌파 예상)\n⚡ 10~30분 내 방향성 돌파 가능성 ↑"
            
            if closes[-1] > (upper[-1] + lower[-1]) / 2:
                movement = '상승 돌파'
                pattern_desc = "삼각 수렴(상승 돌파 예상)\n⬆️ 10~30분 내 상승 돌파 가능성 ↑"
            else:
                movement = '하락 돌파'
                pattern_desc = "삼각 수렴(하락 돌파 예상)\n⬇️ 10~30분 내 하락 돌파 가능성 ↑"
            
            patterns.append({
                'name': '삼각 수렴',
                'confidence': confidence,
                'timeframe': '10~30분',
                'movement': movement,
                'description': pattern_desc
            })

    # 상승 쐐기 - 조건 보완
    if len(df) >= 10:
        last10 = df.iloc[-10:]
        upper = last10['high'].values
        lower = last10['low'].values
        
        upper_slope = np.mean(np.diff(upper))
        lower_slope = np.mean(np.diff(lower))
        
        if (np.all(np.diff(upper) > 0) and
            np.all(np.diff(lower) > 0) and
            lower_slope > 1.5 * upper_slope):
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
    
    if df.empty or len(df) < 3:
        return buy_score, sell_score, buy_reasons, sell_reasons, latest_rsi
    
    try:
        required_keys = ['HMA3', 'HMA', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
        if not all(key in latest for key in required_keys):
            raise ValueError("필수 기술적 지표 데이터 누락")

        # 매수 조건
        if latest['HMA3'] > latest['HMA']:
            buy_score += 3
            buy_reasons.append("HMA3 > HMA (3점)")
        
        # RSI 조건
        if 30 < latest['RSI'] < 70:
            if latest['RSI'] > 45 and latest['RSI'] > df['RSI'].iloc[-2]:
                buy_score += 2
                buy_reasons.append(f"RSI({latest['RSI']:.1f}) > 45 & 상승 (2점)")
        
        # MACD 히스토그램
        if latest['MACD_hist'] > 0 and latest['MACD_hist'] > df.iloc[-2]['MACD_hist']:
            buy_score += 2
            buy_reasons.append("MACD 히스토그램 > 0 & 상승 (2점)")
        
        # 거래량 조건
        if len(df) > 2 and latest['volume'] > df.iloc[-2]['volume'] * 1.2:
            buy_score += 2
            buy_reasons.append("거래량 20% 이상 증가 (2점)")
        
        # 볼린저 밴드 조건
        if latest['close'] < latest['BB_lower'] * 1.01:
            buy_score += 2
            buy_reasons.append("가격 < BB 하한선 근접 (2점)")

        # 매도 조건
        if latest['HMA3'] < latest['HMA']:
            sell_score += 3
            sell_reasons.append("HMA3 < HMA (3점)")
        
        # RSI 조건
        if latest['RSI'] > 70:
            sell_score += 3
            sell_reasons.append(f"RSI({latest['RSI']:.1f}) > 70 (3점)")
        
        # MACD 히스토그램
        if latest['MACD_hist'] < 0 and latest['MACD_hist'] < df.iloc[-2]['MACD_hist']:
            sell_score += 2
            sell_reasons.append("MACD 히스토그램 < 0 & 하락 (2점)")
        
        # 거래량 조건
        if len(df) > 2 and latest['volume'] < df.iloc[-2]['volume'] * 0.8:
            sell_score += 2
            sell_reasons.append("거래량 20% 이상 감소 (2점)")
        
        # 볼린저 밴드 조건
        if latest['close'] > latest['BB_upper'] * 0.99:
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
    
    # 중복 패턴 체크 (15분 이내 동일 패턴 방지)
    duplicate = False
    for p in st.session_state.pattern_history:
        if (p['coin'] == coin and 
            p['name'] == pattern['name'] and 
            (now - p['alert_time']).total_seconds() < 900):
            duplicate = True
            break
    
    if not duplicate:
        st.session_state.pattern_history.append(pattern_record)
        alert_msg = f"🔔 [{coin}] 패턴 감지: {pattern['description']} (가격: {alert_price:,.1f})"
        st.session_state.alerts.append(alert_msg)

# ---------------------- 초기 설정 ----------------------
st.set_page_config(layout="wide", page_title="Crypto Dashboard", page_icon="🚀")

# 형광 녹색 스타일
DOS_GREEN = "#39FF14"
DOS_BG = "#000000"
DOS_RED = "#FF2222"
DOS_BLUE = "#00BFFF"
DOS_ORANGE = "#FF9900"
DOS_PURPLE = "#BF00FF"

@st.cache_data
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-color: rgba(0, 0, 0, 0.9);
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# 배경 이미지 설정
if os.path.exists('moka2.jpg'):
    set_background('moka2.jpg')
else:
    st.warning("배경 이미지 파일을 찾을 수 없습니다. 기본 배경을 사용합니다.")

# 커스텀 스타일 적용
st.markdown(
    f"""
    <style>
    :root {{
        --primary: {DOS_GREEN};
        --secondary: {DOS_BLUE};
        --danger: {DOS_RED};
        --warning: {DOS_ORANGE};
        --info: {DOS_PURPLE};
    }}
    
    body, .stApp {{
        background-color: {DOS_BG} !important;
        color: {DOS_GREEN} !important;
        font-family: 'Consolas', 'Courier New', monospace;
    }}
    
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.85);
        z-index: -1;
    }}
    
    /* 헤더 스타일 */
    .st-emotion-cache-1avcm0n {{
        background: rgba(0, 0, 0, 0.8) !important;
        border-bottom: 1px solid {DOS_GREEN};
    }}
    
    /* 사이드바 스타일 */
    .st-emotion-cache-6qob1r {{
        background: rgba(0, 0, 0, 0.8) !important;
        border-right: 1px solid {DOS_GREEN};
    }}
    
    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 5px;
        margin-bottom: 15px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: rgba(0, 0, 0, 0.7) !important;
        color: {DOS_GREEN} !important;
        border: 1px solid {DOS_GREEN} !important;
        border-radius: 5px;
        padding: 8px 15px;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: rgba(57, 255, 20, 0.2) !important;
        color: {DOS_GREEN} !important;
        border-bottom: 2px solid {DOS_GREEN} !important;
        transform: translateY(-2px);
        box-shadow: 0 0 10px rgba(57, 255, 20, 0.5);
    }}
    
    /* 입력 요소 스타일 */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {{
        background: rgba(0, 0, 0, 0.7) !important;
        color: {DOS_GREEN} !important;
        border: 1px solid {DOS_GREEN} !important;
    }}
    
    .stButton>button {{
        background: rgba(0, 0, 0, 0.7) !important;
        color: {DOS_GREEN} !important;
        border: 1px solid {DOS_GREEN} !important;
        border-radius: 5px;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{
        background: rgba(57, 255, 20, 0.2) !important;
        box-shadow: 0 0 10px rgba(57, 255, 20, 0.5);
    }}
    
    /* 테이블 스타일 */
    .data-table {{
        width: 100%;
        border-collapse: collapse;
        background: rgba(0, 0, 0, 0.7);
        color: {DOS_GREEN};
        border: 2px solid {DOS_GREEN};
        margin: 15px 0;
        font-size: 14px;
    }}
    
    .data-table th, .data-table td {{
        border: 1px solid {DOS_GREEN};
        padding: 10px 12px;
        text-align: center;
    }}
    
    .data-table th {{
        background-color: rgba(0, 50, 0, 0.5);
        font-weight: bold;
        position: sticky;
        top: 0;
    }}
    
    .data-table tr:hover {{
        background-color: rgba(57, 255, 20, 0.1);
    }}
    
    .positive {{
        color: #00FF00;
        font-weight: bold;
    }}
    
    .negative {{
        color: {DOS_RED};
        font-weight: bold;
    }}
    
    /* 차트 타이틀 */
    .chart-title {{
        font-size: 20px;
        font-weight: bold;
        margin: 20px 0 10px;
        color: {DOS_GREEN};
        border-bottom: 2px solid {DOS_GREEN};
        padding-bottom: 8px;
        text-align: center;
    }}
    
    /* 알림 박스 */
    .alert-box {{
        padding: 12px 15px; 
        background: rgba(26, 26, 26, 0.8); 
        border-left: 4px solid {DOS_GREEN};
        border-radius: 5px;
        margin: 12px 0;
        font-size: 14px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    
    /* 패턴 알림 */
    .pattern-alert {{
        border: 1px solid {DOS_GREEN};
        border-radius: 8px;
        padding: 12px;
        margin: 12px 0;
        background: rgba(0, 0, 0, 0.7);
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }}
    
    /* 신호 표시 */
    .signal-box {{
        background: rgba(0, 0, 0, 0.7);
        border: 1px solid {DOS_GREEN};
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }}
    
    .buy-signal {{
        color: #00FF00 !important;
        font-weight: bold;
        font-size: 18px;
    }}
    
    .sell-signal {{
        color: {DOS_RED} !important;
        font-weight: bold;
        font-size: 18px;
    }}
    
    /* 반응형 조정 */
    @media (max-width: 768px) {{
        .stTabs [data-baseweb="tab"] {{
            padding: 6px 10px;
            font-size: 12px;
        }}
        
        .data-table th, .data-table td {{
            padding: 6px 8px;
            font-size: 12px;
        }}
        
        .chart-title {{
            font-size: 16px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# 메인 타이틀
st.markdown(f"<h1 style='text-align:center;color:{DOS_GREEN};font-family:Consolas,monospace;margin-top:10px;'>🔥 If you're not desperate, don't even think about it!</h1>", unsafe_allow_html=True)
st.markdown(f"<h2 style='text-align:center;color:{DOS_GREEN};font-family:Consolas,monospace;margin-bottom:30px;'>🔥 Live Cryptocurrency Analytics Dashboard</h2>", unsafe_allow_html=True)
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
MAIN_COIN = 'KRW-STX'

# ---------------------- 데이터 함수 (성능 개선) ----------------------
@st.cache_data(ttl=10, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get_current_prices():
    try:
        res = requests.get(f"https://api.upbit.com/v1/ticker?markets={','.join(markets)}", timeout=7)
        if res.status_code == 200:
            return {x['market']: x for x in res.json()}
        else:
            st.error(f"가격 조회 실패: {res.status_code}")
            return {}
    except Exception as e:
        st.error(f"가격 조회 오류: {str(e)}")
        return {}

@st.cache_data(ttl=30, show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
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
        df['volume'] = df['volume'].astype(float)

        # 기술적 지표 계산
        if len(df) > 20:
            df['HL2'] = (df['high'] + df['low']) / 2
            df['HMA'] = hma(df['HL2'])
            df['HMA3'] = hma3(df['HL2'])
            df['RSI'] = rsi(df['close'])
            df['MACD_line'], df['Signal_line'], df['MACD_hist'] = macd(df['close'])
            df['BB_ma'], df['BB_upper'], df['BB_lower'] = bollinger_bands(df['close'])
            df['CCI'] = cci(df)
            df['ATR'] = atr(df)
        return df
    except Exception as e:
        st.error(f"{market} 데이터 처리 오류: {str(e)}")
        return pd.DataFrame()

# ---------------------- 차트 생성 함수 (최적화) ----------------------
def create_coin_chart(market, selected_tf):
    coin = market.split('-')[1]
    df = fetch_ohlcv(market, selected_tf)
    if df.empty or len(df) < 20:
        return None, 0, 0, [], []
    
    latest = df.iloc[-1]
    current_price = latest.get('close', 0)
    
    # 신호 점수 계산
    buy_score, sell_score, buy_reasons, sell_reasons, _ = calculate_signal_score(df, latest)
    
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
        name="Price",
        increasing_line_color='red',
        decreasing_line_color='blue'
    ), row=1, col=1)
    
    # 현재가 표시
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
    
    # 기술적 지표
    if 'HMA' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'], 
            y=df['HMA'], 
            name='HMA', 
            line=dict(color='yellow', width=2)
        ), row=1, col=1)
        
    if 'HMA3' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'], 
            y=df['HMA3'], 
            name='HMA3', 
            line=dict(color='magenta', width=2)
        ), row=1, col=1)
        
    if 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'], 
            y=df['BB_upper'], 
            name='BB Upper', 
            line=dict(color='gray', dash='dot', width=1)
        ), row=1, col=1)
        
    if 'BB_lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'], 
            y=df['BB_lower'], 
            name='BB Lower', 
            line=dict(color='gray', dash='dot', width=1)
        ), row=1, col=1)
    
    # 거래량
    colors = np.where(df['close'] >= df['open'], 'red', 'blue')
    fig.add_trace(go.Bar(
        x=df['datetime'], 
        y=df['volume'], 
        name='Volume',
        marker_color=colors,
        opacity=0.7
    ), row=1, col=1, secondary_y=True)
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'], 
            y=df['RSI'], 
            name='RSI', 
            line=dict(color='purple', width=2)
        ), row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="red", row=2, col=1, annotation_text="30")
        fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1, annotation_text="70")
    
    # MACD
    if 'MACD_hist' in df.columns:
        colors = np.where(df['MACD_hist'] >= 0, 'red', 'blue')
        fig.add_trace(go.Bar(
            x=df['datetime'], 
            y=df['MACD_hist'], 
            name='Histogram',
            marker_color=colors,
            opacity=0.7
        ), row=3, col=1)
        
    if 'MACD_line' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'], 
            y=df['MACD_line'], 
            name='MACD', 
            line=dict(color='yellow', width=2)
        ), row=3, col=1)
        
    if 'Signal_line' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['datetime'], 
            y=df['Signal_line'], 
            name='Signal', 
            line=dict(color='magenta', width=2)
        ), row=3, col=1)

    fig.update_layout(
        height=700,
        title={
            'text': f"{coin} 차트 ({timeframes[selected_tf]})",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24, 'color': DOS_GREEN}
        },
        xaxis_rangeslider_visible=False,
        margin=dict(t=60, b=40, l=40, r=40),
        showlegend=True,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0.5)',
        paper_bgcolor='rgba(0,0,0,0.5)',
        font=dict(color=DOS_GREEN),
        hovermode="x unified"
    )
    
    # X축 범위 설정 (최근 50개 데이터만 표시)
    if len(df) > 50:
        fig.update_xaxes(range=[df['datetime'].iloc[-50], df['datetime'].iloc[-1]])

    return fig, buy_score, sell_score, buy_reasons, sell_reasons

# ---------------------- 사이드바 설정 ----------------------
with st.sidebar:
    st.markdown(f"<div style='color:{DOS_GREEN};font-family:Consolas,monospace;'>", unsafe_allow_html=True)
    st.header("⚙️ 제어 패널")
    selected_tf = st.selectbox('차트 주기', list(timeframes.keys()), format_func=lambda x: timeframes[x])
    
    st.subheader("💰 투자 현황")
    prices = get_current_prices()
    st.session_state.cached_prices = prices
    
    stx_holding = default_holdings['KRW-STX']
    stx_price = prices.get('KRW-STX', {}).get('trade_price', 0)
    current_value = stx_holding * stx_price
    profit = current_value - TOTAL_INVESTMENT
    profit_percent = (profit / TOTAL_INVESTMENT) * 100 if TOTAL_INVESTMENT else 0
    profit_emoji = "🔻" if profit < 0 else "🟢"
    profit_color = DOS_BLUE if profit < 0 else DOS_RED
    
    st.markdown(f"<div style='color:{DOS_GREEN};'>총 투자금액</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{DOS_GREEN};font-size:22px;'>{TOTAL_INVESTMENT:,.0f} 원</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div style='color:{DOS_GREEN};'>STX 자산가치</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{profit_color};font-size:22px;'>{current_value:,.0f} 원</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div style='color:{profit_color};font-size:18px;'>{profit_emoji} {profit:+,.0f} 원 ({profit_percent:+.2f}%)</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div style='color:{DOS_GREEN};'>STX 보유량</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{DOS_GREEN};font-size:22px;'>{stx_holding:,.2f} EA</div>", unsafe_allow_html=True)
    
    st.subheader("🔔 알림 설정")
    telegram_enabled = st.checkbox("텔레그램 알림 활성화")
    table_alert_interval = st.number_input("테이블 알림 주기(분)", min_value=1, value=10)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- 메인 화면 초기화 ----------------------
def init_session_state():
    session_vars = {
        'alerts': [],
        'last_alert_time': {},
        'last_table_alert_time': datetime.min,
        'pattern_history': [],
        'detected_patterns': {market: [] for market in markets},
        'cached_prices': {},
        'selected_coin': 'STX',
        'minute_counter': 0
    }
    
    for key, value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ---------------------- 코인 비교 테이블 (통합) ----------------------
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
        
        # 기술적 지표 및 패턴 감지
        df = fetch_ohlcv(market, selected_tf)
        buy_score, sell_score = 0, 0
        latest_rsi = 0
        detected_patterns = []

        if not df.empty and len(df) >= 3:
            try:
                latest = df.iloc[-1]
                if 'RSI' in df.columns:
                    latest_rsi = latest['RSI']
                if 'HMA3' in df.columns and 'HMA' in df.columns:
                    buy_score, sell_score, _, _, _ = calculate_signal_score(df, latest)
                
                # 패턴 감지
                detected_patterns = detect_chart_patterns(df)
                st.session_state.detected_patterns[market] = detected_patterns
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
        
        # 패턴 정보 요약
        pattern_summary = ""
        if detected_patterns:
            for pattern in detected_patterns[:2]:
                pattern_summary += f"<div>• {pattern['name']}(<b>{pattern['confidence']}%</b>) "
                if pattern['movement'] == '상승':
                    pattern_summary += "⬆️</div>"
                elif pattern['movement'] == '하락':
                    pattern_summary += "⬇️</div>"
                else:
                    pattern_summary += "↔️</div>"
        
        change_color = DOS_BLUE if change_rate < 0 else DOS_RED
        change_emoji = "🔻" if change_rate < 0 else "🟢"
        buy_color = "#00FF00" if buy_score >= 7 else DOS_GREEN
        sell_color = DOS_RED if sell_score >= 7 else DOS_GREEN
        
        compare_data.append({
            '코인명': coin,
            '시세': f"{price:,.1f} 원" if price > 0 else "-",
            'RSI': f"{latest_rsi:.1f}" if latest_rsi > 0 else "-",
            '매수신호': f"<span style='color:{buy_color}'>매수({buy_score}/10)</span>",
            '매도신호': f"<span style='color:{sell_color}'>매도({sell_score}/10)</span>",
            '등락률': f"<span style='color:{change_color}'>{change_emoji} {change_rate:+.2f}%</span>",
            '패턴 예측': pattern_summary if pattern_summary else "-",
            '보유수량': f"{qty:,.2f}",
            '대체가능수량': f"{replace_qty:,.2f}" if market != base_market else "-",
            '차이수량': f"{diff_qty:+,.2f}" if market != base_market else "-",
            '평가금액': f"{value:,.0f}",
            '대체평가액': f"{replace_value:,.0f}" if market != base_market else "-"
        })

    return pd.DataFrame(compare_data)

# ---------------------- 탭 구성 ----------------------
tab1, tab2 = st.tabs(["📊 코인 분석 대시보드", "📈 개별 코인 차트 분석"])

with tab1:
    # 코인 비교 테이블
    st.markdown(f"<div class='chart-title'>코인 분석 테이블</div>", unsafe_allow_html=True)
    df_compare = generate_coin_table()
    
    # HTML 테이블로 스타일 적용
    table_html = f"""
    <table class="data-table">
        <thead>
            <tr>
                <th>코인명</th>
                <th>시세</th>
                <th>RSI</th>
                <th>매수신호</th>
                <th>매도신호</th>
                <th>등락률</th>
                <th>패턴 예측</th>
                <th>보유수량</th>
                <th>대체가능수량</th>
                <th>차이수량</th>
                <th>평가금액</th>
                <th>대체평가액</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in df_compare.iterrows():
        table_html += "<tr>"
        
        # 코인명
        table_html += f'<td>{row["코인명"]}</td>'
        
        # 시세
        table_html += f'<td>{row["시세"]}</td>'
        
        # RSI
        rsi_value = float(row["RSI"]) if row["RSI"] != "-" else 0
        rsi_class = "positive" if rsi_value < 30 else "negative" if rsi_value > 70 else ""
        table_html += f'<td class="{rsi_class}">{row["RSI"]}</td>'
        
        # 매수신호
        table_html += f'<td>{row["매수신호"]}</td>'
        
        # 매도신호
        table_html += f'<td>{row["매도신호"]}</td>'
        
        # 등락률
        change_class = "positive" if '+' in row["등락률"] else "negative" if '-' in row["등락률"] else ""
        table_html += f'<td class="{change_class}">{row["등락률"]}</td>'
        
        # 패턴 예측
        table_html += f'<td>{row["패턴 예측"]}</td>'
        
        # 보유수량
        table_html += f'<td>{row["보유수량"]}</td>'
        
        # 대체가능수량
        table_html += f'<td>{row["대체가능수량"]}</td>'
        
        # 차이수량
        diff_class = "positive" if isinstance(row["차이수량"], str) and row["차이수량"].startswith('+') else "negative" if isinstance(row["차이수량"], str) and row["차이수량"].startswith('-') else ""
        table_html += f'<td class="{diff_class}">{row["차이수량"]}</td>'
        
        # 평가금액
        table_html += f'<td>{row["평가금액"]}</td>'
        
        # 대체평가액
        table_html += f'<td>{row["대체평가액"]}</td>'
        
        table_html += "</tr>"
    
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    
    
    st.subheader("📊 코인 분석 대시보드 (통합)")
    
    # RSI 비교 차트
    st.markdown(f"<div class='chart-title'>RSI 비교 차트 ({timeframes[selected_tf]})</div>", unsafe_allow_html=True)
    fig_rsi = make_subplots(rows=1, cols=1)
    rsi_time_window = timedelta(days=3)
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
                    name=f'{coin}',
                    line=dict(width=3)
                ))
    
    if len(fig_rsi.data) > 0:
        fig_rsi.update_layout(
            height=400,
            title=None,
            yaxis_title="RSI",
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0.5)',
            paper_bgcolor='rgba(0,0,0,0.5)',
            font=dict(color=DOS_GREEN),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="과매도")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="과매수")
        fig_rsi.add_annotation(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text="<b>RSI 비교 차트</b>", showarrow=False,
            font=dict(size=16, color=DOS_GREEN)
        )
        st.plotly_chart(fig_rsi, use_container_width=True)
    else:
        st.warning("RSI 데이터가 없습니다")
    
    
    # 테이블 알림 생성
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
                f"패턴 예측: {row['패턴 예측'].replace('• ', '  - ')}\n"
                f"보유량: {row['보유수량']}\n"
                f"평가금액: {row['평가금액']}원\n"
            )
            if row['코인명'].upper() in ['HBAR', 'DOGE']:
                alert_msg += (
                    f"대체 가능 수량: {row['대체가능수량']}\n"
                    f"차이 수량: {row['차이수량']}\n"
                )
            alert_msg += "\n"
        
        st.session_state.last_table_alert_time = now
        if telegram_enabled:
            send_telegram_alert(alert_msg.strip())
        if len(st.session_state.alerts) > 20:
            st.session_state.alerts = st.session_state.alerts[-20:]
        st.session_state.alerts.append(alert_msg)

# ---------------------- 개별 코인 분석 ----------------------
with tab2:
    st.subheader("📈 개별 코인 차트 분석")
    
    # 코인 선택
    coin_options = [market.split('-')[1] for market in markets]
    selected_coin = st.selectbox(
        "분석할 코인 선택", 
        coin_options, 
        index=coin_options.index(st.session_state.selected_coin),
        key='coin_selector'
    )
    market = f"KRW-{selected_coin}"
    
    # 차트 생성
    fig, buy_score, sell_score, buy_reasons, sell_reasons = create_coin_chart(market, selected_tf)
    
    if fig:
        # 신호 상태 표시
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='signal-box'>", unsafe_allow_html=True)
            st.markdown(f"<div class='buy-signal'>매수 신호: {buy_score}/10</div>", unsafe_allow_html=True)
            for reason in buy_reasons:
                st.markdown(f"<div>✓ {reason}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
                
        with col2:
            st.markdown(f"<div class='signal-box'>", unsafe_allow_html=True)
            st.markdown(f"<div class='sell-signal'>매도 신호: {sell_score}/10</div>", unsafe_allow_html=True)
            for reason in sell_reasons:
                st.markdown(f"<div>✓ {reason}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 메인 코인(STX)에 대해서만 패턴 알림
        if market == MAIN_COIN and st.session_state.detected_patterns.get(market):
            pattern_alerts = []
            now = datetime.now()
            for pattern in st.session_state.detected_patterns[market]:
                timeframe_str = timeframes[selected_tf]
                current_price = prices.get(market, {}).get('trade_price', 0)
                alert_msg = f"🚨🚨 [{selected_coin} {timeframe_str}차트] {current_price:,.1f}원 // {pattern['description']} ({now.strftime('%H:%M:%S')})"
                
                # 중복 알림 방지
                last_pattern_alert_key = f"{selected_coin}_{pattern['name']}_pattern"
                last_alert_time = st.session_state.last_alert_time.get(last_pattern_alert_key, datetime.min)
                
                if (now - last_alert_time) > timedelta(minutes=10):
                    if telegram_enabled:
                        send_telegram_alert(alert_msg)
                    st.session_state.last_alert_time[last_pattern_alert_key] = now
                    pattern_alerts.append(alert_msg)
            
            st.session_state.alerts.extend(pattern_alerts)
            process_pattern_alerts(selected_coin, pattern, current_price)

# ---------------------- 실시간 패턴 업데이트 ----------------------
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

# ---------------------- 사이드바 패턴 정보 표시 ----------------------
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
            diff_color = DOS_RED if price_diff < 0 else DOS_BLUE
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
            change_color = DOS_RED if change_percent < 0 else DOS_BLUE
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

# ---------------------- 실시간 알림 출력 ----------------------
st.subheader("🔔 분석알림 (RSI 및 패턴 포함)")
if st.session_state.alerts:
    # 알림 개수 관리 (최대 20개)
    if len(st.session_state.alerts) > 20:
        st.session_state.alerts = st.session_state.alerts[-20:]
    
    for alert in reversed(st.session_state.alerts[-10:]):
        # 패턴 알림 강조
        if "패턴 감지" in alert:
            st.markdown(f"<div class='alert-box' style='border-left: 4px solid {DOS_ORANGE};'>🚨 {alert}</div>", unsafe_allow_html=True)
        # 매수/매도 신호 강조
        elif "매수" in alert or "매도" in alert:
            border_color = "#00FF00" if "매수" in alert else DOS_RED
            st.markdown(f"<div class='alert-box' style='border-left: 4px solid {border_color};'>📢 {alert}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='alert-box'>ℹ️ {alert}</div>", unsafe_allow_html=True)
else:
    st.info("최근 알림이 없습니다")

# ---------------------- 푸터 영역 ----------------------
st.markdown("---")
st.markdown(f"<div style='text-align:center;color:{DOS_GREEN};margin-top:30px;'>"
            f"<b>MIWOONI Crypto Dashboard</b> | Real-time Analytics | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"</div>", unsafe_allow_html=True)

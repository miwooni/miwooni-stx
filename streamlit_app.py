import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import joblib
from streamlit_autorefresh import st_autorefresh

# MinMaxScaler를 직접 구현 (sklearn 의존성 제거)
class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min = None
        self.data_max = None
        self.scale = None
        self.min = None
        
    def fit(self, X):
        self.data_min = np.min(X, axis=0)
        self.data_max = np.max(X, axis=0)
        self.scale = (self.feature_range[1] - self.feature_range[0]) / (self.data_max - self.data_min + 1e-7)
        self.min = self.feature_range[0] - self.data_min * self.scale
        
    def transform(self, X):
        return self.min + X * self.scale
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X):
        return (X - self.min) / self.scale

# 비밀번호 설정
PASSWORD = "Fudfud8080@"

# 세션 상태 초기화
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 인증 처리
if not st.session_state.authenticated:
    st.title("🔐 모카 간식비 만들기 프로젝트")
    password = st.text_input("비밀번호를 입력하세요:", type="password")
    if password == PASSWORD:
        st.session_state.authenticated = True
        st.rerun()
    elif password != "":
        st.error("❌ 비밀번호가 틀렸습니다.")
    st.stop()

# ✅ 반드시 첫 번째 Streamlit 명령어로 설정
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed"  # 사이드바 초기 숨김
)

# ---------------------- 세션 상태 초기화 ----------------------
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {}
if 'ma_cross_alerts' not in st.session_state:
    st.session_state.ma_cross_alerts = {}
if 'total_investment' not in st.session_state:
    st.session_state.total_investment = 58500000
if 'telegram_bot_token' not in st.session_state:
    st.session_state.telegram_bot_token = "7545404316:AAHMdayWZjwEZmZwd5JrKXPDn5wUQfivpTw"
if 'telegram_chat_id' not in st.session_state:
    st.session_state.telegram_chat_id = "7890657899"

# 자동 갱신 설정 (5초 간격)
st_autorefresh(interval=30000, key="auto_refresh")

# ---------------------- 갤럭시 기기 최적화 반응형 디자인 ----------------------
DOS_GREEN = "#39FF14"
DOS_BG = "#000000"
st.markdown(
    f"""
    <style>
    body, .stApp {{
        background-color: {DOS_BG} !important;
        font-family: 'Nanum Gothic', sans-serif;
    }}
    .dos-green, .stMarkdown, .stDataFrame, .stTable, .stText, .stSubheader, .stHeader, .stTitle, .stMetric, .stSidebar, .stSidebarContent, .stSidebar .stTextInput, .stSidebar .stNumberInput, .stSidebar .stCheckbox, .stSidebar .stSelectbox, .stSidebar .stMetric {{
        color: {DOS_GREEN} !important;
        background-color: {DOS_BG} !important;
    }}
    .stMetric label, .stMetric div, .stMetric span {{
        color: {DOS_GREEN} !important;
    }}
    
    /* 갤럭시탭 S8 울트라 (2800x1752) & 갤럭시 Z 플립3 (1080x2640) 대응 */
    @media (max-width: 600px) {{
        .stSelectbox, .stNumberInput, .stTextInput, .stButton > button {{
            font-size: 16px !important;
            padding: 12px !important;
        }}
        .stAlert {{
            padding: 8px !important;
            font-size: 14px;
        }}
        .table-wrapper {{
            overflow-x: auto;
            font-size: 12px;
        }}
        h1 {{ font-size: 1.8rem !important; }}
        h2 {{ font-size: 1.5rem !important; }}
        h3 {{ font-size: 1.3rem !important; }}
        .plotly-chart {{ height: 300px !important; }}
    }}
    
    @media (min-width: 601px) and (max-width: 1200px) {{
        .stSelectbox, .stNumberInput, .stTextInput, .stButton > button {{
            font-size: 18px !important;
            padding: 14px !important;
        }}
        .stAlert {{
            padding: 10px !important;
            font-size: 16px;
        }}
        .table-wrapper {{
            overflow-x: auto;
            font-size: 14px;
        }}
        h1 {{ font-size: 2.2rem !important; }}
        h2 {{ font-size: 1.8rem !important; }}
        h3 {{ font-size: 1.5rem !important; }}
        .plotly-chart {{ height: 400px !important; }}
    }}
    
    /* 스크롤바 디자인 */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    ::-webkit-scrollbar-track {{
        background: {DOS_BG};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {DOS_GREEN};
        border-radius: 5px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #2ECC71;
    }}
    
    /* 탭 디자인 개선 */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 12px 24px;
        border-radius: 8px;
        background-color: #111;
        transition: all 0.3s;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #222;
        box-shadow: 0 0 10px {DOS_GREEN};
    }}
    
    /* 테이블 디자인 개선 */
    .dataframe {{
        border: 1px solid {DOS_GREEN} !important;
    }}
    .dataframe th {{
        background-color: #222 !important;
        color: {DOS_GREEN} !important;
    }}
    .dataframe tr:nth-child(even) {{
        background-color: #111 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- 메인 타이틀 ----------------------
st.markdown(f"<h3 style='color:{DOS_GREEN};background:{DOS_BG};font-family:Consolas,monospace;'>🔥 If you're not desperate, don't even think about it!</h3>", unsafe_allow_html=True)
st.markdown(f"<h5 style='color:{DOS_GREEN};background:{DOS_BG};font-family:Consolas,monospace;'>🔥 우리 모카 간식비를 벌자!!!</h5>", unsafe_allow_html=True)

# ---------------------- 전역 변수 ----------------------
default_holdings = {
    'KRW-STX': 13702.725635935,
    'KRW-HBAR': 62216.22494886,
    'KRW-DOGE': 61194.37067502,
}
markets = list(default_holdings.keys())
timeframes = {1: '1분', 3: '3분', 5: '5분', 15: '15분', 60: '60분', 240: '240분'}

# ---------------------- 단순화된 예측 모델 ----------------------
def simple_predict(df, period=5):
    """이동평균 기반 단순 예측"""
    if len(df) < 10:
        return []
    
    # 최근 5개 종가 기반으로 평균 예측
    last_prices = df['close'].tail(5).values
    avg_price = np.mean(last_prices)
    
    # 예측값 생성 (평균값에서 약간의 변동 추가)
    predictions = []
    for i in range(1, 6):
        # 랜덤한 변동 추가 (최근 변동성의 50% 수준)
        volatility = df['close'].pct_change().std() * 0.5
        prediction = avg_price * (1 + np.random.uniform(-volatility, volatility))
        predictions.append(prediction)
    
    return predictions

# ---------------------- 기술적 지표 계산 함수 ----------------------
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

# ---------------------- 신호 점수 계산 함수 ----------------------
def calculate_signal_score(df, latest, pred_prices=None):
    buy_score, sell_score, risk_score = 0, 0, 0
    buy_reasons, sell_reasons, risk_reasons = [], [], []
    latest_rsi = latest.get('RSI', 0) if not df.empty else 0

    # 지표별 가중치
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
        if latest['MACD_hist'] > 0 and latest['MACD_hist'] > df.iloc[-2]['MACD_hist']:
            buy_score += WEIGHTS['macd'] * 1.5
            buy_reasons.append("MACD 상승추세")
        elif latest['MACD_hist'] > 0:
            buy_score += WEIGHTS['macd'] * 1
            buy_reasons.append("MACD > 0")
            
        if latest['MACD_hist'] < 0 and latest['MACD_hist'] < df.iloc[-2]['MACD_hist']:
            sell_score += WEIGHTS['macd'] * 1.5
            sell_reasons.append("MACD 하락추세")
        elif latest['MACD_hist'] < 0:
            sell_score += WEIGHTS['macd'] * 1
            sell_reasons.append("MACD < 0")

        # 거래량
        vol_ma = df['volume'].rolling(10).mean().iloc[-1]
        if latest['volume'] > vol_ma * 1.5:
            if latest['close'] > latest['open']:
                buy_score += WEIGHTS['volume'] * 2
                buy_reasons.append("거래량 급증 & 양봉")
            else:
                sell_score += WEIGHTS['volume'] * 2
                sell_reasons.append("거래량 급증 & 음봉")
        elif latest['volume'] > df.iloc[-2]['volume']:
            buy_score += WEIGHTS['volume'] * 1
            buy_reasons.append("거래량 증가")
        elif latest['volume'] < df.iloc[-2]['volume']:
            sell_score += WEIGHTS['volume'] * 1
            sell_reasons.append("거래량 감소")

        # 볼린저 밴드 돌파
        bb_position = (latest['close'] - latest['BB_lower']) / (latest['BB_upper'] - latest['BB_lower'])
        if bb_position < 0.2:
            buy_score += WEIGHTS['bb_breakout'] * (1 + (0.2 - bb_position) * 5)
            buy_reasons.append(f"BB 하단 근접 ({bb_position:.2f})")
        if bb_position > 0.8:
            sell_score += WEIGHTS['bb_breakout'] * (1 + (bb_position - 0.8) * 5)
            sell_reasons.append(f"BB 상단 근접 ({bb_position:.2f})")

        # AI 예측 반영
        if pred_prices is not None and len(pred_prices) > 0:
            predicted = pred_prices[0]  # 첫 번째 예측값 사용
            diff_percent = (predicted - latest['close']) / latest['close'] * 100
            
            if diff_percent > 1.0:
                buy_score += WEIGHTS['ai_predict'] * 1.5
                buy_reasons.append(f"AI 강력 상승 예측 (+{diff_percent:.1f}%)")
            elif diff_percent > 0.5:
                buy_score += WEIGHTS['ai_predict'] * 1
                buy_reasons.append(f"AI 상승 예측 (+{diff_percent:.1f}%)")
                
            if diff_percent < -1.0:
                sell_score += WEIGHTS['ai_predict'] * 1.5
                sell_reasons.append(f"AI 강력 하락 예측 ({diff_percent:.1f}%)")
            elif diff_percent < -0.5:
                sell_score += WEIGHTS['ai_predict'] * 1
                sell_reasons.append(f"AI 하락 예측 ({diff_percent:.1f}%)")

        # 리스크: 최근 변동성 급등 (가격 변화 or 거래량 폭등)
        if abs(df['close'].pct_change().iloc[-1]) > 0.05:
            risk_score += 2
            risk_reasons.append("단기 급등/급락 경고")
        elif abs(df['close'].pct_change().iloc[-1]) > 0.03:
            risk_score += 1
            risk_reasons.append("변동성 증가")
            
        if latest['volume'] > df['volume'].mean() * 5:
            risk_score += 2
            risk_reasons.append("비정상 거래량")
        elif latest['volume'] > df['volume'].mean() * 3:
            risk_score += 1
            risk_reasons.append("거래량 급증")

    except Exception as e:
        buy_reasons.append(f"오류 발생: {str(e)}")
        sell_reasons.append(f"오류 발생: {str(e)}")

    return buy_score, sell_score, risk_score, buy_reasons, sell_reasons, risk_reasons, latest_rsi

# ---------------------- 데이터 함수 ----------------------
@st.cache_data(ttl=10)
def get_current_prices():
    try:
        res = requests.get(f"https://api.upbit.com/v1/ticker?markets={','.join(markets)}", timeout=5)
        return {x['market']:x for x in res.json()}
    except:
        return {market: {'trade_price': 0, 'signed_change_rate': 0} for market in markets}

@st.cache_data(ttl=30)
def fetch_ohlcv(market, timeframe, count=300):
    try:
        url = f"https://api.upbit.com/v1/candles/minutes/{timeframe}"
        params = {'market': market, 'count': count}
        res = requests.get(url, params=params, timeout=10)
        data = res.json()
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)[::-1]
        df = df[['candle_date_time_kst','opening_price','high_price','low_price','trade_price','candle_acc_trade_volume']]
        df.columns = ['datetime','open','high','low','close','volume']
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 0 값 필터링
        df = df[df['close'] > 0]
        df = df[df['volume'] > 0]

        if len(df) < 10:
            return pd.DataFrame()

        df['HL2'] = (df['high'] + df['low']) / 2
        df['HMA'] = hma(df['HL2'])
        df['HMA3'] = hma3(df['HL2'])
        df['Signal'] = np.where(df['HMA3'] > df['HMA'], '매수', '매도')
        df['RSI'] = rsi(df['close'])
        df['MACD_line'], df['Signal_line'], df['MACD_hist'] = macd(df['close'])
        df['BB_ma'], df['BB_upper'], df['BB_lower'] = bollinger_bands(df['close'])
        df['CCI'] = cci(df)
        df['ATR'] = atr(df)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_ma']
        df['volume_momentum'] = df['volume'] / df['volume'].shift(1)
        
        return df.dropna()
    except Exception as e:
        st.error(f"데이터 가져오기 오류: {str(e)}")
        return pd.DataFrame()

# ---------------------- 코인 테이블 생성 함수 ----------------------
def generate_coin_table(selected_tf):
    signal_scores = {} 
    base_market = 'KRW-STX'
    base_qty = default_holdings[base_market]
    base_price_data = prices.get(base_market, {'trade_price': 0, 'signed_change_rate': 0})
    base_price = base_price_data['trade_price']
    base_krw = base_qty * base_price * 0.9995

    compare_data = []
    for market in markets:
        coin = market.split('-')[1]
        price_data = prices.get(market, {'trade_price': 0, 'signed_change_rate': 0})
        price = price_data['trade_price']
        change_rate = price_data['signed_change_rate'] * 100
        qty = default_holdings[market]
        value = price * qty
        
        df = fetch_ohlcv(market, selected_tf)
        buy_score, sell_score = 0, 0
        latest_rsi = 0
        ai_pred = 0

        if not df.empty and len(df) >= 2:
            try:
                latest = df.iloc[-1]
                required_cols = ['HMA', 'HMA3', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
                if all(col in df.columns for col in required_cols):
                    # AI 예측값 가져오기
                    ai_preds = simple_predict(df)
                    if ai_preds:
                        ai_pred = ai_preds[0]
                    buy_score, sell_score, _, _, _, _, latest_rsi = calculate_signal_score(df, latest, pred_prices=ai_preds)
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
        
        change_color = "#FF6B6B" if change_rate < 0 else "#51CF66"
        change_emoji = "🔻" if change_rate < 0 else "🟢"
        buy_color = "#51CF66" if buy_score >= 7 else "gray"
        sell_color = "#FF6B6B" if sell_score >= 7 else "gray"
        
        # AI 예측 방향성
        ai_direction = ""
        if ai_pred > 0:
            ai_diff = ((ai_pred - price) / price) * 100
            if ai_diff > 1:
                ai_direction = f"<span style='color:#51CF66; font-weight:bold;'>▲ {ai_diff:.1f}%</span>"
            elif ai_diff > 0:
                ai_direction = f"<span style='color:#A9E34B;'>△ {ai_diff:.1f}%</span>"
            elif ai_diff < -1:
                ai_direction = f"<span style='color:#FF6B6B; font-weight:bold;'>▼ {-ai_diff:.1f}%</span>"
            else:
                ai_direction = f"<span style='color:#FFA94D;'>▽ {-ai_diff:.1f}%</span>"
        
        compare_data.append({
            '코인명': coin,
            '시세': f"{price:,.1f} 원",
            'RSI': f"{latest_rsi:.1f}",
            'AI예측': ai_direction,
            '매수신호': f"<span style='color:{buy_color}'>매수({buy_score:.1f}/10)</span>",
            '매도신호': f"<span style='color:{sell_color}'>매도({sell_score:.1f}/10)</span>",
            '등락률': f"<span style='color:{change_color}'>{change_emoji} {change_rate:+.2f}%</span>",
            '보유수량': qty,
            '대체가능수량': replace_qty if market != base_market else "-",
            '차이수량': diff_qty if market != base_market else "-",
            '평가금액': value,
            '대체평가액': replace_value if market != base_market else "-"
        })

    df_compare = pd.DataFrame(compare_data)
    df_compare['차이수량'] = df_compare['차이수량'].apply(lambda x: float(x) if x != "-" else x)
    df_compare['대체가능수량'] = df_compare['대체가능수량'].apply(lambda x: float(x) if x != "-" else x)
    return df_compare

def format_number(x):
    if isinstance(x, (int, float)):
        if x == "-": return x
        return f"{x:,.2f}" if abs(x) < 10000 else f"{x:,.0f}"
    return x

# ---------------------- 사용자 입력 ----------------------
with st.sidebar:
    st.subheader("📈 차트 주기(분) 선택")
    selected_tf = st.selectbox("차트 주기", list(timeframes.keys()), 
                              format_func=lambda x: timeframes[x], key="selected_tf", index=2)
    
    st.header("🔑 사용자 설정")
    with st.expander("텔레그램 설정"):
        telegram_enabled = st.checkbox("✉️ 텔레그램 알림 활성화", value=False)
        table_alert_interval = st.number_input("테이블 알림 주기(분)", min_value=1, value=10)
        st.session_state.telegram_bot_token = st.text_input(
            "텔레그램 봇 토큰",
            value=st.session_state.telegram_bot_token
        )
        st.session_state.telegram_chat_id = st.text_input(
            "텔레그램 채팅 ID",
            value=st.session_state.telegram_chat_id
        )
    
    with st.expander("보유 코인 설정"):
        for market in ['KRW-STX', 'KRW-HBAR', 'KRW-DOGE']:
            coin = market.split('-')[1]
            default_holdings[market] = st.number_input(
                f"{coin} 보유량",
                value=default_holdings.get(market, 0.0),
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
    
    with st.expander("AI 설정"):
        st.subheader("🤖 AI 예측 설정")
        pred_horizon = st.slider("예측 기간 (봉)", 1, 10, 5)
        ai_confidence = st.slider("신뢰도 임계값", 50, 100, 80)
        st.info("이동평균 기반 단순 예측 모델 사용 중")

# ---------------------- 텔레그램 알림 함수 ----------------------
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

# ---------------------- 메인 콘텐츠 ----------------------
prices = get_current_prices()

tab1, tab2 = st.tabs(["📊 코인 비교 테이블", "📈 상세 차트 분석"])

with tab1:
    st.subheader(f"📊 코인 비교 테이블 ({timeframes[selected_tf]} 차트 기준)")
    df_compare = generate_coin_table(selected_tf)
    
    def diff_qty_color(val):
        try:
            v = float(val)
            if v > 0: return "color:#FF6B6B"
            elif v < 0: return "color:#51CF66"
            else: return f"color:{DOS_GREEN}"
        except: return f"color:{DOS_GREEN}"

    def change_rate_color(val):
        try:
            import re
            match = re.search(r'([+\-]?\d+(\.\d+)?)', str(val))
            num = float(match.group(1)) if match else 0
            if '-' in str(val): return "color:#51CF66"
            elif '+' in str(val): return "color:#FF6B6B"
            elif num > 0: return "color:#FF6B6B"
            elif num < 0: return "color:#51CF66"
            else: return f"color:{DOS_GREEN}"
        except: return f"color:{DOS_GREEN}"

    styled = (
        df_compare.style.format({
            '보유수량': format_number,
            '대체가능수량': format_number,
            '차이수량': lambda x: f"{x:+,.0f}" if isinstance(x, (int, float)) else x,
            '평가금액': format_number,
            '대체평가액': format_number,
            'RSI': lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else str(x)
        })
        .map(diff_qty_color, subset=['차이수량'])
        .map(change_rate_color, subset=['등락률'])
        .map(lambda _: 'text-align: center')
        .set_properties(**{'background-color': '#111', 'color': DOS_GREEN})
    )
    st.markdown('<div class="table-wrapper">' + styled.to_html(escape=False, index=False) + '</div>', unsafe_allow_html=True)
    
    # 알림 내역 표시
    st.subheader("📢 실시간 알림 내역")
    alert_history = st.session_state.get("alerts", [])
    if alert_history:
        for alert in reversed(alert_history[-10:]):
            st.markdown(
                f"<div style='background:#222;padding:12px;margin:8px 0;border-radius:8px;font-size:16px;color:#39FF14;border-left:4px solid #51CF66;'>📢 {alert}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("현재 알림 내역이 없습니다")

with tab2:
    st.subheader("📈 차트 분석")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_coin = st.selectbox("코인 선택", [m.split('-')[1] for m in markets], key="chart_coin")
        market_for_coin = [m for m in markets if m.split('-')[1] == selected_coin][0]
        
        # 기술적 지표 선택
        st.subheader("기술적 지표 설정")
        show_sma = st.checkbox("이동평균선(SMA)", value=True)
        show_bollinger = st.checkbox("볼린저 밴드", value=True)
        show_macd = st.checkbox("MACD", value=True)
        show_rsi = st.checkbox("RSI", value=True)
        show_volume = st.checkbox("거래량", value=True)
    
    with col2:
        df_ma = fetch_ohlcv(market_for_coin, selected_tf)
        
        if not df_ma.empty:
            # AI 예측값 가져오기
            ai_preds = simple_predict(df_ma)
            pred_times = []
            if ai_preds:
                last_time = df_ma['datetime'].iloc[-1]
                pred_times = [last_time + timedelta(minutes=selected_tf*(i+1)) for i in range(len(ai_preds))]
            
            # 차트 생성
            fig = make_subplots(
                rows=3, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2],
                specs=[[{"secondary_y": True}], [{}], [{}]]
            )
            
            # 가격 차트
            fig.add_trace(go.Scatter(
                x=df_ma['datetime'], y=df_ma['close'],
                name='종가', line=dict(color='#339AF0', width=2),
                hovertemplate='%{y:,.0f}원<extra></extra>'
            ), row=1, col=1)
            
            # 캔들스틱 차트
            fig.add_trace(go.Candlestick(
                x=df_ma['datetime'],
                open=df_ma['open'],
                high=df_ma['high'],
                low=df_ma['low'],
                close=df_ma['close'],
                name='캔들',
                increasing_line_color='#51CF66', 
                decreasing_line_color='#FF6B6B'
            ), row=1, col=1)
            
            # 이동평균선
            if show_sma:
                ma_periods = [5, 10, 20, 60]
                colors = ['#FCC419', '#FF922B', '#E64980', '#BE4BDB']
                for period, color in zip(ma_periods, colors):
                    col_name = f'SMA{period}'
                    df_ma[col_name] = df_ma['close'].rolling(period).mean()
                    fig.add_trace(go.Scatter(
                        x=df_ma['datetime'], y=df_ma[col_name],
                        name=f'SMA{period}', line=dict(width=1.5, color=color),
                        visible='legendonly' if period > 20 else True
                    ), row=1, col=1)
            
            # 볼린저 밴드
            if show_bollinger:
                fig.add_trace(go.Scatter(
                    x=df_ma['datetime'], y=df_ma['BB_upper'],
                    name='BB 상한', line=dict(width=1, color='#868E96', dash='dash'),
                    fill='tonexty', fillcolor='rgba(134, 142, 150, 0.1)'
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df_ma['datetime'], y=df_ma['BB_ma'],
                    name='BB 중간', line=dict(width=1, color='#868E96', dash='dot'),
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df_ma['datetime'], y=df_ma['BB_lower'],
                    name='BB 하한', line=dict(width=1, color='#868E96', dash='dash'),
                    fill='tonexty', fillcolor='rgba(134, 142, 150, 0.1)'
                ), row=1, col=1)
            
            # HMA 신호
            fig.add_trace(go.Scatter(
                x=df_ma['datetime'], y=df_ma['HMA'],
                name='HMA', line=dict(width=2, color='#F783AC')
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df_ma['datetime'], y=df_ma['HMA3'],
                name='HMA3', line=dict(width=2, color='#5C7CFA')
            ), row=1, col=1)
            
            # AI 예측값
            if ai_preds and pred_times:
                fig.add_trace(go.Scatter(
                    x=pred_times, y=ai_preds,
                    name='AI 예측',
                    mode='markers+lines',
                    line=dict(width=3, color='#FFD43B', dash='dot'),
                    marker=dict(size=10, color='#FFD43B', symbol='diamond'),
                    hovertemplate='AI 예측: %{y:,.0f}원<extra></extra>'
                ), row=1, col=1)
                
                for i, (t, pred) in enumerate(zip(pred_times, ai_preds)):
                    fig.add_annotation(
                        x=t, y=pred,
                        text=f"{pred:,.0f}",
                        showarrow=True,
                        arrowhead=1,
                        ax=0, ay=-30,
                        bgcolor="rgba(255,212,59,0.8)",
                        font=dict(color="#000", size=12)
                    )
            
            # 거래량 차트
            if show_volume:
                fig.add_trace(go.Bar(
                    x=df_ma['datetime'], y=df_ma['volume'],
                    name='거래량', marker_color='#4DABF7',
                    opacity=0.7
                ), row=2, col=1)
            
            # MACD 차트
            if show_macd:
                fig.add_trace(go.Scatter(
                    x=df_ma['datetime'], y=df_ma['MACD_line'],
                    name='MACD', line=dict(width=1.5, color='#20C997')
                ), row=3, col=1)
                fig.add_trace(go.Scatter(
                    x=df_ma['datetime'], y=df_ma['Signal_line'],
                    name='Signal', line=dict(width=1.5, color='#FA5252')
                ), row=3, col=1)
                
                # MACD 히스토그램
                colors = ['#20C997' if val >= 0 else '#FA5252' for val in df_ma['MACD_hist']]
                fig.add_trace(go.Bar(
                    x=df_ma['datetime'], y=df_ma['MACD_hist'],
                    name='Histogram', marker_color=colors,
                    opacity=0.6
                ), row=3, col=1)
            
            # RSI 차트
            if show_rsi:
                fig.add_trace(go.Scatter(
                    x=df_ma['datetime'], y=df_ma['RSI'],
                    name='RSI', line=dict(width=1.5, color='#CC5DE8'),
                    yaxis='y2'
                ), row=1, col=1)
                fig.add_hline(y=30, line=dict(width=1, dash='dash', color='#51CF66'), row=1, col=1)
                fig.add_hline(y=70, line=dict(width=1, dash='dash', color='#FF6B6B'), row=1, col=1)
            
            # 레이아웃 설정
            fig.update_layout(
                title=f"{selected_coin} 차트 분석 ({timeframes[selected_tf]} 봉)",
                height=800,
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(rangeslider=dict(visible=False)),
                xaxis2=dict(showticklabels=True),
                xaxis3=dict(showticklabels=True),
                yaxis2=dict(title="거래량"),
                yaxis3=dict(title="MACD"),
                yaxis=dict(title="가격 (원)"),
                yaxis4=dict(
                    title="RSI",
                    overlaying="y",
                    side="right",
                    range=[0, 100],
                    showgrid=False
                ),
                margin=dict(l=50, r=50, t=80, b=50),
                plot_bgcolor=DOS_BG,
                paper_bgcolor=DOS_BG,
                font=dict(color=DOS_GREEN)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 신호 점수 표시
            if not df_ma.empty:
                latest = df_ma.iloc[-1]
                buy_score, sell_score, risk_score, buy_reasons, sell_reasons, risk_reasons, latest_rsi = calculate_signal_score(
                    df_ma, latest, ai_preds
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("매수 신호 점수", f"{buy_score}/20", delta="강한 매수" if buy_score > 12 else "매수 관심")
                with col2:
                    st.metric("매도 신호 점수", f"{sell_score}/20", delta="강한 매도" if sell_score > 12 else "매도 관심", delta_color="inverse")
                with col3:
                    st.metric("리스크 점수", f"{risk_score}/4", delta="위험 높음" if risk_score > 2 else "안정적", 
                             delta_color="normal" if risk_score <= 2 else "off")
                
                # 신호 이유 표시
                with st.expander("매수 신호 이유"):
                    if buy_reasons:
                        for reason in buy_reasons:
                            st.markdown(f"- ✅ {reason}")
                    else:
                        st.info("매수 신호가 없습니다")
                
                with st.expander("매도 신호 이유"):
                    if sell_reasons:
                        for reason in sell_reasons:
                            st.markdown(f"- ⚠️ {reason}")
                    else:
                        st.info("매도 신호가 없습니다")
                
                with st.expander("리스크 요인"):
                    if risk_reasons:
                        for reason in risk_reasons:
                            st.markdown(f"- ⚠️ {reason}")
                    else:
                        st.info("특별한 리스크 요인이 없습니다")
        else:
            st.warning(f"{selected_coin} 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")

# ---------------------- 이동평균선 교차 알림 로직 ----------------------
for market in markets:
    coin = market.split('-')[1]
    df = fetch_ohlcv(market, selected_tf)
    if len(df) > 200:
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

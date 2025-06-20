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
    if len(df) < 7: return patterns

    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    atr_vals = df['ATR'].values

    # W 패턴 (거래량 + 변동성 필터)
    if len(df) >= 5:
        first_low = lows[-5]
        second_low = lows[-1]
        if (abs(first_low - second_low) < 0.02*(first_low+second_low)/2 and
            volumes[-1] > np.mean(volumes[-5:])*1.2 and
            atr_vals[-1] > np.mean(atr_vals[-5:])*0.7):
            patterns.append({
                'name': 'W 패턴',
                'confidence': 80,
                'timeframe': '5~15분',
                'movement': '상승',
                'description': "W 패턴 (확률 80%)\n🚀 5~15분 후 상승 확률 ↑"
            })

    # 역 헤드앤숄더 (다중 조건)
    if len(df) >= 7:
        head_low = lows[-4]
        left_shoulder = lows[-6]
        right_shoulder = lows[-2]
        neckline = (highs[-7] + highs[-5] + highs[-3] + highs[-1])/4
        if (left_shoulder > head_low < right_shoulder and
            abs(left_shoulder - right_shoulder) < 0.03*head_low and
            df['close'].iloc[-1] > neckline and
            volumes[-4] > np.mean(volumes)*1.5):
            patterns.append({
                'name': '역 헤드앤숄더',
                'confidence': 85,
                'timeframe': '10~30분',
                'movement': '상승',
                'description': "역 헤드앤숄더 (확률 85%)\n🚀 10~30분 후 상승 확률 ↑"
            })

    # 상승 깃발 (추세 지속)
    if len(df) >= 8:
        last8 = df.iloc[-8:]
        range_cond = last8['high'] - last8['low']
        if (np.all(range_cond[:3] > 2*range_cond.mean()) and
            np.all(np.diff(last8['close'][3:]) < 0.02*last8['close'].mean())):
            patterns.append({
                'name': '상승 깃발패턴',
                'confidence': 75,
                'timeframe': '10~20분',
                'movement': '상승',
                'description': "상승 깃발패턴(추세 지속)\n🚀 10~20분 후 추가 상승 가능성 ↑"
            })

    # 하락 쐐기 (상승 반전)
    if len(df) >= 10:
        last10 = df.iloc[-10:]
        upper = last10['high'].rolling(3).max().dropna()
        lower = last10['low'].rolling(3).min().dropna()
        if (np.all(np.diff(upper) < 0) and
            np.all(np.diff(lower) < 0)):
            patterns.append({
                'name': '하락 쐐기',
                'confidence': 70,
                'timeframe': '15~30분',
                'movement': '반등',
                'description': "하락 쐐기(상승 반전 예측)\n🚀 15~30분 후 반등 가능성 ↑"
            })

    # 컵 앤 핸들 (상승 지속)
    if len(df) >= 20:
        last20 = df.iloc[-20:]
        cup = last20[:15]['low'].values
        handle = last20[15:]['low'].values
        if (np.all(cup[0] > cup[1:7]) and
            np.all(cup[-5:] > cup[7]) and
            np.all(handle[-3:] > handle[0])):
            patterns.append({
                'name': '컵 앤 핸들',
                'confidence': 80,
                'timeframe': '30~60분',
                'movement': '강한 상승',
                'description': "컵 앤 핸들(상승 지속)\n🚀 30~60분 후 강한 상승 신호"
            })

    # 삼각 수렴 (상승/하락 브레이크아웃)
    if len(df) >= 10:
        last10 = df.iloc[-10:]
        upper = last10['high'].values
        lower = last10['low'].values
        if (np.all(np.diff(upper) < 0) and
            np.all(np.diff(lower) > 0)):
            patterns.append({
                'name': '삼각 수렴',
                'confidence': 75,
                'timeframe': '10~30분',
                'movement': '방향성 돌파',
                'description': "삼각 수렴(추세 모멘텀)\n⚡ 10~30분 내 방향성 돌파 가능성 ↑"
            })

    return patterns

def calculate_signal_score(df, latest):
    buy_score = 0
    sell_score = 0
    buy_reasons = []
    sell_reasons = []
    latest_rsi = latest.get('RSI', 0) if not df.empty else 0
    
    if df.empty or len(df) < 2:
        return buy_score, sell_score, buy_reasons, sell_reasons, latest_rsi
    
    try:
        required_keys = ['HMA3', 'HMA', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
        if not all(key in latest for key in required_keys):
            raise ValueError("필수 기술적 지표 데이터 누락")

        # 매수 조건
        if latest['HMA3'] > latest['HMA']:
            buy_score += 3
            buy_reasons.append("HMA3 > HMA (3점)")
        if latest['RSI'] > 30:
            buy_score += 2
            buy_reasons.append(f"RSI({latest['RSI']:.1f}) > 40 (2점)")
        if latest['MACD_hist'] > 0 and latest['MACD_hist'] > df.iloc[-2]['MACD_hist']:
            buy_score += 1
            buy_reasons.append("MACD 히스토그램 > 0 (1점)")
        if len(df) > 1 and latest['volume'] > df.iloc[-2]['volume']:
            buy_score += 2
            buy_reasons.append("거래량 증가 (2점)")
        if latest['close'] < latest['BB_lower']:
            buy_score += 2
            buy_reasons.append("가격 < BB 하한선 (2점)")

        # 매도 조건
        if latest['HMA3'] < latest['HMA']:
            sell_score += 3
            sell_reasons.append("HMA3 < HMA (3점)")
        if latest['RSI'] > 70:
            sell_score += 2
            sell_reasons.append(f"RSI({latest['RSI']:.1f}) > 70 (2점)")
        if latest['MACD_hist'] < 0:
            sell_score += 1
            sell_reasons.append("MACD 히스토그램 < 0 (1점)")
        if len(df) > 1 and latest['volume'] < df.iloc[-2]['volume']:
            sell_score += 2
            sell_reasons.append("거래량 감소 (2점)")
        if latest['close'] > latest['BB_upper']:
            sell_score += 2
            sell_reasons.append("가격 > BB 상한선 (2점)")
            
    except Exception as e:
        st.error(f"신호 계산 오류: {str(e)}")
        buy_reasons = [f"오류 발생: {str(e)}"]
        sell_reasons = [f"오류 발생: {str(e)}"]
    
    return buy_score, sell_score, buy_reasons, sell_reasons, latest_rsi

# 패턴 알림 처리 함수 추가
def process_pattern_alerts(coin, pattern_desc, alert_price):
    now = datetime.now()
    # 패턴 이력에 추가
    pattern = {
        'coin': coin,
        'pattern': pattern_desc,
        'alert_time': now,
        'alert_price': alert_price,
        'current_price': alert_price,  # 초기값은 알림 가격
        'completed': False,
        'end_price': None
    }
    st.session_state.pattern_history.append(pattern)
    # 알림 메시지 추가
    alert_msg = f"🔔 [{coin}] 패턴 감지: {pattern_desc} (가격: {alert_price:,.1f})"
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
        df['CCI'] = cci(df)
        df['ATR'] = atr(df)
        return df
    except:
        return pd.DataFrame()

# ---------------------- 사이드바 설정 ----------------------
with st.sidebar:
    st.markdown(f"<div style='color:{DOS_GREEN};font-family:Consolas,monospace;'>", unsafe_allow_html=True)
    st.header("⚙️ 제어 패널")
    selected_tf = st.selectbox('차트 주기', list(timeframes.keys()), format_func=lambda x: timeframes[x])
    
    st.subheader("💰 투자 현황")
    prices = get_current_prices()
    stx_holding = default_holdings['KRW-STX']
    stx_price = prices['KRW-STX']['trade_price']
    current_value = stx_holding * stx_price
    profit = current_value - TOTAL_INVESTMENT
    profit_percent = (profit / TOTAL_INVESTMENT) * 100
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

# ---------------------- 메인 화면 ----------------------
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = {}
if 'last_table_alert_time' not in st.session_state:
    st.session_state.last_table_alert_time = datetime.min
if 'pattern_history' not in st.session_state:
    st.session_state.pattern_history = []
if 'detected_patterns' not in st.session_state:
    st.session_state.detected_patterns = {}

# ---------------------- 코인 비교 테이블 ----------------------
def generate_coin_table():
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

        if not df.empty and len(df) >= 2:
            try:
                latest = df.iloc[-1]
                required_cols = ['HMA', 'HMA3', 'RSI', 'MACD_hist', 'volume', 'close', 'BB_upper', 'BB_lower']
                if all(col in df.columns for col in required_cols):
                    buy_score, sell_score, _, _, latest_rsi = calculate_signal_score(df, latest)
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
            '평가금액': value,
            '대체평가액': replace_value if market != base_market else "-"
        })

    df_compare = pd.DataFrame(compare_data)
    # 차이수량/대체가능수량은 '-'가 아닌 경우만 float 변환
    df_compare['차이수량'] = df_compare['차이수량'].apply(lambda x: float(x) if x != "-" else x)
    df_compare['대체가능수량'] = df_compare['대체가능수량'].apply(lambda x: float(x) if x != "-" else x)
    return df_compare

def format_number(x):
    if isinstance(x, (int, float)):
        if x == "-": return x
        return f"{x:,.2f}" if abs(x) < 10000 else f"{x:,.0f}"
    return x

# ---------------------- 탭 구성 ----------------------
tab1, tab2, tab3 = st.tabs(["📊 코인 비교 테이블", "📈 코인 분석", "🔮 코인 예측"])

with tab1:
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

    # 등락률 컬럼: -파랑, +빨강 (HTML 태그 내 숫자 추출, 부호 우선)
    def change_rate_color(val):
        try:
            import re
            match = re.search(r'([+\-]?\d+(\.\d+)?)', str(val))
            num = float(match.group(1)) if match else 0
            if '-' in str(val):
                return "color:#00BFFF"
            elif '+' in str(val):
                return "color:#FF2222"
            elif num > 0:
                return "color:#FF2222"
            elif num < 0:
                return "color:#00BFFF"
            else:
                return f"color:{DOS_GREEN}"
        except Exception:
            return f"color:{DOS_GREEN}"

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
    st.markdown(styled.to_html(escape=False, index=False), unsafe_allow_html=True)

with tab2:
    # 2코인 분석 탭: RSI 비교 차트 + 예측 가격 분석
    st.subheader("📈 RSI 비교 차트")
    fig_rsi = make_subplots(rows=1, cols=1)
    rsi_time_window = timedelta(days=14)
    now_time = datetime.now()
    for market in markets:
        coin = market.split('-')[1]
        df = fetch_ohlcv(market, selected_tf)
        if not df.empty:
            df_recent = df[df['datetime'] >= now_time - rsi_time_window]
            fig_rsi.add_trace(go.Scatter(
                x=df_recent['datetime'],
                y=df_recent['RSI'],
                name=f'{coin} RSI',
                line=dict(width=2)
            ))
    fig_rsi.update_layout(
        height=400,
        title=f"RSI 비교 ({timeframes[selected_tf]} 차트)",
        yaxis_title="RSI",
        hovermode="x unified"
    )
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    st.plotly_chart(fig_rsi, use_container_width=True)

# ---------------------- 테이블 알림 생성 ----------------------
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

# ---------------------- 개별 코인 분석 ----------------------
for market in markets:
    coin = market.split('-')[1]
    df = fetch_ohlcv(market, selected_tf)
    if df.empty:
        st.warning(f"{coin} 차트 데이터 불러오기 실패")
        continue

    latest = df.iloc[-1]
    current_price = latest['close']
    prev_close = df.iloc[-2]['close'] if len(df) > 1 else current_price
    delta = current_price - prev_close
    delta_percent = (delta / prev_close) * 100 if prev_close else 0

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
        row_heights=[0.6, 0.2, 0.2],
        specs=[[{"secondary_y": True}], [{}], [{}]]
    )
    
    fig.add_trace(go.Candlestick(
        x=df['datetime'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name="Price"), row=1, col=1)
    
    fig.add_hline(y=current_price, line_dash="solid", line_color="red", row=1, col=1,
                 annotation_text=f"현재가: {current_price:,.1f}", 
                 annotation_position="top left",
                 annotation_font_color="blue",
                 annotation_font_size=14,
                 annotation_xanchor='center')
    
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['HMA'], name='HMA', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['HMA3'], name='HMA3', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_upper'], name='BB Upper', line=dict(color='gray', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['BB_lower'], name='BB Lower', line=dict(color='gray', dash='dot')), row=1, col=1)
    fig.add_trace(go.Bar(x=df['datetime'], y=df['volume'], name='Volume',
                       marker_color=np.where(df['close'] > df['open'], 'green', 'red')),
               row=1, col=1, secondary_y=True)
    
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)

    fig.add_trace(go.Bar(x=df['datetime'], y=df['MACD_hist'], name='Histogram',
                       marker_color=np.where(df['MACD_hist'] > 0, 'green', 'red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['MACD_line'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Signal_line'], name='Signal', line=dict(color='orange')), row=3, col=1)

    fig.update_layout(
        height=800,
        title=f"{coin} 차트 ({selected_tf})",
        xaxis_rangeslider_visible=False,
        margin=dict(t=40, b=40),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # 메인 코인(STX)에 대해서만 패턴 감지 및 알림
    if market == MAIN_COIN:
        detected_patterns = detect_chart_patterns(df)
        pattern_alerts = []
        for pattern in detected_patterns:
            timeframe_str = timeframes[selected_tf]
            alert_msg = f"🚨🚨 [{coin} {timeframe_str}차트]{current_price:,.1f} 원 // 강력한 패턴 경고 - {pattern['description']} 발현! ({now.strftime('%H:%M:%S')})"
            last_pattern_alert_key = f"{coin}_{pattern['name']}_pattern"
            last_alert_time, last_alert_price = st.session_state.last_alert_time.get(last_pattern_alert_key, (datetime.min, None))
            price_change = 0
            if last_alert_price is not None and last_alert_price != 0:
                price_change = abs((current_price - last_alert_price) / last_alert_price) * 100
            if (now - last_alert_time) > timedelta(minutes=5) or price_change >= 1:
                if telegram_enabled:
                    send_telegram_alert(alert_msg)
                st.session_state.last_alert_time[last_pattern_alert_key] = (now, current_price)
                pattern_alerts.append(alert_msg)
        st.session_state.alerts.extend(pattern_alerts)
        st.session_state.detected_patterns[coin] = detected_patterns

# ---------------------- 패턴 알림 처리 로직 (메인 코인만) ----------------------
if MAIN_COIN in st.session_state.detected_patterns:
    coin = MAIN_COIN.split('-')[1]
    detected_patterns = st.session_state.detected_patterns[coin]
    for pattern in detected_patterns:
        alert_price = prices[MAIN_COIN]['trade_price']
        process_pattern_alerts(coin, pattern['description'], alert_price)

# ---------------------- 코인 예측 탭 ----------------------
with tab3:
    st.subheader("🔮 코인 예측 패턴 분석")
    
    prediction_data = []
    for market in markets:
        coin = market.split('-')[1]
        df = fetch_ohlcv(market, selected_tf)
        if df.empty:
            continue
            
        patterns = detect_chart_patterns(df)
        for pattern in patterns:
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
                return f"background-color: #FF2222; color: white; font-weight: bold;"
            return "background-color: #FFFF00; color: black;"
        
        # hide_index() 제거
        styled = (
            df_predictions.style
            .applymap(color_strength, subset=['신호 강도'])
            .format({'확률(%)': '{:.0f}%'})
        )
        st.markdown(styled.to_html(escape=False), unsafe_allow_html=True)
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
            pattern['end_price'] = pattern['current_price']  # 현재 가격을 종가로 기록

# ---------------------- 사이드바 패턴 정보 표시 (메인 코인만) ----------------------
with st.sidebar:
    st.subheader("🔮 패턴 예측 분석")
    st.markdown(f"<div style='color:{DOS_GREEN};font-weight:bold;'>🔔 활성 패턴 알림</div>", 
                unsafe_allow_html=True)
    active_patterns = [p for p in st.session_state.pattern_history 
                      if not p.get('completed') and p['coin'] == MAIN_COIN.split('-')[1]]
    
    if not active_patterns:
        st.markdown(f"<div style='color:{DOS_GREEN};'>- 활성 알림 없음</div>", unsafe_allow_html=True)
    else:
        for pattern in active_patterns:
            elapsed_min = (datetime.now() - pattern['alert_time']).seconds // 60
            st.markdown(
                f"<div style='color:{DOS_GREEN};margin-bottom:10px;'>"
                f"▫️ <b>{pattern['coin']}</b> ({pattern['pattern']})<br>"
                f"- 예측시간: {pattern['alert_time'].strftime('%H:%M:%S')}<br>"
                f"- 예측가격: {pattern['alert_price']:,.1f}원<br>"
                f"- 현재가격: {pattern['current_price']:,.1f}원<br>"
                f"- 경과시간: {elapsed_min}분"
                f"</div>",
                unsafe_allow_html=True
            )
    
    st.markdown(f"<div style='color:{DOS_GREEN};font-weight:bold;margin-top:20px;'>📊 최근 패턴 결과</div>", 
                unsafe_allow_html=True)
    completed_patterns = [p for p in st.session_state.pattern_history 
                         if p.get('completed') and p['coin'] == MAIN_COIN.split('-')[1]][-3:]
    if not completed_patterns:
        st.markdown(f"<div style='color:{DOS_GREEN};'>- 완료된 알림 없음</div>", unsafe_allow_html=True)
    else:
        for pattern in completed_patterns:
            change_percent = (pattern['end_price'] - pattern['alert_price']) / pattern['alert_price'] * 100
            change_color = "#FF2222" if change_percent < 0 else "#00BFFF"
            st.markdown(
                f"<div style='color:{DOS_GREEN};margin-bottom:15px;border:1px solid {DOS_GREEN};padding:10px;border-radius:5px;'>"
                f"▫️ <b>{pattern['coin']}</b> ({pattern['pattern']})<br>"
                f"- 예측시간: {pattern['alert_time'].strftime('%H:%M:%S')}<br>"
                f"- 예측가격: {pattern['alert_price']:,.1f}원<br>"
                f"- 종료가격: {pattern['end_price']:,.1f}원<br>"
                f"- 변동률: <span style='color:{change_color};'>{change_percent:+.2f}%</span><br>"
                f"- 경과시간: 15분"
                f"</div>",
                unsafe_allow_html=True
            )

# ---------------------- 실시간 알림 출력 ----------------------
st.subheader("🔔 분석알림 (RSI 및 패턴 포함)")
for alert in reversed(st.session_state.alerts[-10:]):
    st.markdown(f"""<div style='padding:10px; background:#2b2b2b; border-radius:5px; margin:5px 0;'>{alert}</div>""", unsafe_allow_html=True)

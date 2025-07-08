import streamlit as st
import pandas as pd
import numpy as np
import math
import json
import time
import asyncio
import aiohttp
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# 색상 정의
DOS_GREEN = "#00FF00"
DOS_BG = "#000000"

# 상수 정의
INCH_TO_MM = 25.4
PI = math.pi

# 비밀번호 설정
PASSWORD = "1234"  # 실제 사용시 변경

# 피치 유형 정의
PITCH_TYPES = {
    '10" (6피치+4피치)': {'pitch': 10, 'density': 2},
    '8"': {'pitch': 8, 'density': 1},
    '6"': {'pitch': 6, 'density': 1}
}

# 부품 목록
PARTS = [
    {'id': 'main_chain', 'name': '메인 체인', 'unit': 'm', 'default_purchase': 21000, 'default_sale': 27500},
    {'id': 'link', 'name': '연결 링크', 'unit': '개', 'default_purchase': 6500, 'default_sale': 11000},
    {'id': 'troly', 'name': '트로리', 'unit': '조', 'default_purchase': 3500, 'default_sale': 5000},
    {'id': 'bolt', 'name': '렌치볼트 (M6x35L)', 'unit': '개', 'default_purchase': 200, 'default_sale': 250},
    {'id': 'nut', 'name': '나이론 너트 (M6)', 'unit': '개', 'default_purchase': 100, 'default_sale': 150},
    {'id': 'assembly', 'name': '트로리 조립 인건비', 'unit': 'm', 'default_purchase': 5500, 'default_sale': 6000},
    {'id': 'shackle1', 'name': '1차 샤클 (킬링샤클)', 'unit': '개', 'default_purchase': 10000, 'default_sale': 15000},
    {'id': 'shackle2', 'name': '2차 샤클 (내장샤클)', 'unit': '개', 'default_purchase': 12000, 'default_sale': 18000},
    {'id': 'pack_shackle', 'name': '팩샤클', 'unit': '개', 'default_purchase': 8000, 'default_sale': 12000},
    {'id': 'airchilling_shackle', 'name': '에어칠링 샤클', 'unit': '개', 'default_purchase': 15000, 'default_sale': 22000},
]

# 샤클 규격 옵션
SHACKLE_SIZES = ["6\"", "8\"", "10\""]
SHACKLE_PRICES = {
    '6"': {'purchase': 10000, 'sale': 15000},
    '8"': {'purchase': 12000, 'sale': 18000},
    '10"': {'purchase': 15000, 'sale': 22000},
}

# 초기 데이터프레임 (캐싱 적용)
@st.cache_data
def init_data():
    return pd.DataFrame([{
        '품목': p['name'], '규격': p['id'], '단위': p['unit'],
        '수량': 0, '구매 단가': p['default_purchase'], '판매 단가': p['default_sale'],
        '구매 금액': 0, '판매 금액': 0
    } for p in PARTS])

# 수량 계산 (캐싱 적용)
@st.cache_data
def calculate_requirements(pitch_type, length):
    pitch_data = PITCH_TYPES.get(pitch_type, {'pitch': 6, 'density': 1})
    pitch_mm = pitch_data['pitch'] * INCH_TO_MM
    density = pitch_data['density']

    troly_per_m = (1000 / pitch_mm) * density
    troly_count = round(troly_per_m * length)

    # 샤클 자동 계산
    shackle_qty = 0
    shackle_size = ""
    if pitch_data['pitch'] == 6:
        shackle_qty = round(length * (1000 / (6 * INCH_TO_MM)))
        shackle_size = '6"'
    elif pitch_data['pitch'] == 8:
        shackle_qty = round(length * (1000 / (8 * INCH_TO_MM)))
        shackle_size = '8"'
    elif pitch_data['pitch'] == 10:
        shackle_qty = round(length * (1000 / (10 * INCH_TO_MM)))
        shackle_size = '10"'

    return {
        'main_chain': length,
        'link': max(1, round(length / 20)),
        'troly': troly_count,
        'bolt': troly_count * 2,
        'nut': troly_count * 2,
        'assembly': length,
        'shackle1': shackle_qty,
        'shackle2': shackle_qty,
        'pack_shackle': 0,
        'airchilling_shackle': 0,
        'shackle_size': shackle_size
    }

# PCD 계산 함수 (벡터화)
def calculate_pcd_vectorized(teeth_count, pitch_inch):
    pitch_mm = pitch_inch * INCH_TO_MM
    return (pitch_mm * teeth_count) / PI

# 스프라켓 계산 함수 (최적화)
def calculate_sprocket_travel(sprocket_pcd, motor_rpm, reduction_ratio):
    sprocket_circumference = sprocket_pcd * math.pi
    travel_per_min = motor_rpm * sprocket_circumference
    travel_per_sec = travel_per_min / 60
    travel_per_sec_reduced = travel_per_sec / reduction_ratio
    
    return {
        "circumference": sprocket_circumference,
        "per_min": travel_per_min,
        "per_sec": travel_per_sec,
        "per_sec_reduced": travel_per_sec_reduced
    }

# 최종 비용 계산 (벡터화)
def calculate_final_costs(df, quote_mode):
    if quote_mode == "internal":
        df['구매 금액'] = df['수량'] * df['구매 단가']
    df['판매 금액'] = df['수량'] * df['판매 단가']
    
    total_sale = df['판매 금액'].sum()
    results = {"total_sale": total_sale, "item_count": len(df)}
    
    if quote_mode == "internal":
        total_purchase = df['구매 금액'].sum()
        total_profit = total_sale - total_purchase
        profit_margin = (total_profit / total_purchase * 100) if total_purchase > 0 else 0
        results.update({
            "total_purchase": total_purchase,
            "total_profit": total_profit,
            "profit_margin": profit_margin
        })
    
    return results

# 고속 Ollama 챗봇 (스트리밍 + 비동기)
async def ollama_chat_async():
    st.header("⚡ 고속 Ollama AI 챗봇")
    
    # 세션 상태 초기화
    if "ollama_messages" not in st.session_state:
        st.session_state.ollama_messages = [
            {"role": "system", "content": "사용자의 질문에 간결하고 정확하게 답변하세요. 답변은 3문장 이내로 요약하세요."}
        ]
    
    # 채팅 기록 표시
    for msg in st.session_state.ollama_messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
    
    # 사용자 입력 처리
    if prompt := st.chat_input("Ollama에게 질문하세요 (엔터를 누르세요)"):
        # 사용자 메시지 추가
        st.session_state.ollama_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt, unsafe_allow_html=True)
        
        # AI 응답 플레이스홀더
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Ollama 스트리밍 API 호출
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": "llama3",  # 더 빠른 모델 사용
                            "messages": st.session_state.ollama_messages,
                            "stream": True,
                            "options": {
                                "temperature": 0.3,  # 창의성 감소
                                "num_ctx": 1024,     # 컨텍스트 크기 축소
                                "num_predict": 256   # 최대 출력 길이 제한
                            }
                        }
                    ) as response:
                        
                        # 스트리밍 응답 처리
                        async for chunk in response.content:
                            if chunk:
                                decoded_chunk = chunk.decode('utf-8')
                                if decoded_chunk.strip():
                                    try:
                                        json_chunk = json.loads(decoded_chunk)
                                        content_chunk = json_chunk.get('message', {}).get('content', '')
                                        if content_chunk:
                                            full_response += content_chunk
                                            message_placeholder.markdown(full_response + "▌")
                                    except json.JSONDecodeError:
                                        pass
                    
                    # 최종 응답 업데이트
                    message_placeholder.markdown(full_response)
                    
                    # 세션 상태 업데이트
                    st.session_state.ollama_messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Ollama 연결 오류: {str(e)}")
                    st.info("Ollama가 실행 중인지 확인해주세요. 설치 가이드: https://ollama.com/")

# Streamlit 앱
def main():
    st.set_page_config(
        page_title="초고속 체인 견적기", 
        page_icon="⚡", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 성능 최적화 CSS
    st.markdown(f"""
    <style>
        /* 기본 스타일 최적화 */
        body, .stApp {{ background-color: {DOS_BG}; color: {DOS_GREEN}; }}
        .st-bq, .st-cb, .st-cd, .st-ce, .st-cf, .st-cg, .st-ch {{ color: {DOS_GREEN} !important; }}
        
        /* 폰트 최적화 */
        * {{ 
            font-family: 'Courier New', monospace !important; 
            font-size: 1.1em !important;
        }}
        
        /* 데이터 테이블 최적화 */
        .stDataFrame {{ 
            font-size: 0.9em !important;
            max-height: 500px;
            overflow: auto;
        }}
        
        /* 채팅창 최적화 */
        .stChatMessage {{
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }}
        .stChatInput {{ 
            position: fixed;
            bottom: 2rem;
            width: calc(100% - 4rem);
            background: {DOS_BG};
            z-index: 100;
        }}
        
        /* 버튼 최적화 */
        .stButton>button {{
            background-color: #333;
            color: {DOS_GREEN};
            border: 1px solid {DOS_GREEN};
            padding: 0.3em 1em;
            font-weight: bold;
        }}
        
        /* 사이드바 최적화 */
        .stSidebar {{
            background-color: #111 !important;
        }}
        
        /* 애니메이션 제거 */
        * {{ 
            transition: none !important; 
            animation: none !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    st.title("⚡ 초고속 오버헤드 트로리 견적 시스템")

    # 세션 상태 초기화
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'quote_mode' not in st.session_state:
        st.session_state.quote_mode = "internal"
    if 'prev_troly' not in st.session_state:
        st.session_state.prev_troly = None
    if 'pitch_type' not in st.session_state:
        st.session_state.pitch_type = '10" (6피치+4피치)'
    if 'chain_length' not in st.session_state:
        st.session_state.chain_length = 20.0

    # 메뉴 선택 (사이드바)
    with st.sidebar:
        st.header("⚡ 고속 메뉴")
        menu_options = [
            "예상견적가", 
            "도계라인 길이 계산", 
            "도계라인 체류시간 계산", 
            "도계라인 속도 계산", 
            "스프라켓/모터 이동 거리 계산",
            "PCD 계산기",
            "Ollama 챗봇"
        ]
        menu = st.radio("계산기 선택", menu_options, index=0)
        
        if menu == "예상견적가":
            st.markdown("---")
            st.header("견적 모드")
            quote_mode = st.radio(
                "모드 선택", 
                ["내부용 (모든 정보 표시)", "고객용 (내부 정보 숨김)"],
                index=0 if st.session_state.quote_mode == "internal" else 1,
                key="quote_mode_selector"
            )
            st.session_state.quote_mode = "internal" if quote_mode == "내부용 (모든 정보 표시)" else "customer"
            
            st.markdown("---")
            st.header("시스템 설정")
            
            pitch_type_options = list(PITCH_TYPES.keys())
            pitch_type = st.selectbox(
                "피치 유형",
                pitch_type_options,
                index=pitch_type_options.index(st.session_state.pitch_type),
                key='pitch_type'
            )
            length = st.number_input("길이 (m)", min_value=1.0, 
                                    value=st.session_state.chain_length, 
                                    step=1.0, key="chain_length", format="%.2f")

        st.markdown("---")
        st.markdown("""
        <div style="color:#39FF14; font-size:0.9em; text-align:center;">
            <span>© 2024 주식회사 티제이젠<br>All Rights Reserved.</span><br>
            <span style="display:block;margin-top:10px;font-weight:bold;">
            ⚡ SUNG JIN KANG
            </span>
        </div>
        """, unsafe_allow_html=True)

    # 메인 화면 처리
    if menu == "도계라인 길이 계산":
        st.header("⚡ 도계라인 길이 계산기")
        with st.form("length_calculator"):
            col1, col2 = st.columns(2)
            speed1 = col1.number_input("도계속도 (수/HR)", value=12500, min_value=1)
            pitch1 = col1.selectbox("샤클 피치 (인치)", [6, 8, 10], index=0)
            time1 = col2.number_input("체류시간 (초)", value=253, min_value=1)
            
            if st.form_submit_button("계산", use_container_width=True):
                length_m = (speed1 * pitch1 * INCH_TO_MM * time1) / (3600 * 1000)
                st.session_state.calculated_length = length_m
                st.success(f"계산된 길이: **{length_m:.2f} m**")
                minutes, seconds = divmod(time1, 60)
                st.info(f"체류시간: {time1}초 ({int(minutes)}분 {int(seconds)}초)")

    elif menu == "도계라인 체류시간 계산":
        st.header("⚡ 도계라인 체류시간 계산기")
        with st.form("time_calculator"):
            col1, col2 = st.columns(2)
            speed2 = col1.number_input("도계속도 (수/HR)", value=12500, min_value=1)
            pitch2 = col1.selectbox("샤클 피치 (인치)", [6, 8, 10], index=0)
            distance2 = col2.number_input("거리 (mm)", value=90000.0, min_value=0.1)
            
            if st.form_submit_button("계산", use_container_width=True):
                time_sec = (distance2 * 3600) / (speed2 * pitch2 * INCH_TO_MM)
                st.success(f"계산된 체류시간: **{time_sec:.2f} 초**")
                minutes, seconds = divmod(time_sec, 60)
                st.info(f"({int(minutes)}분 {int(seconds)}초)")

    elif menu == "도계라인 속도 계산":
        st.header("⚡ 도계라인 속도 계산기")
        with st.form("speed_calculator"):
            col1, col2 = st.columns(2)
            pitch3 = col1.selectbox("샤클 피치 (인치)", [6, 8, 10], index=0)
            time3 = col1.number_input("체류시간 (초)", value=120967, min_value=1)
            distance3 = col2.number_input("거리 (m)", value=88.8, min_value=0.1)
            
            if st.form_submit_button("계산", use_container_width=True):
                speed_hr = (distance3 * 1000 * 3600) / (pitch3 * INCH_TO_MM * time3)
                st.success(f"계산된 도계속도: **{speed_hr:.2f} 수/HR**")
                minutes, seconds = divmod(time3, 60)
                st.info(f"체류시간: {int(minutes)}분 {int(seconds)}초")

    elif menu == "스프라켓/모터 이동 거리 계산":
        st.header("⚡ 스프라켓/모터 이동 거리 계산기")
        with st.form("sprocket_calculator"):
            col1, col2 = st.columns(2)
            sprocket_pcd = col1.number_input("스프라켓 PCD (mm)", value=168.0, min_value=1.0, step=1.0)
            motor_rpm = col1.number_input("모터 RPM", value=1750, min_value=1, step=1)
            reduction_ratio = col2.number_input("감속비", value=60.0, min_value=0.1, step=1.0)
            
            if st.form_submit_button("계산", use_container_width=True):
                results = calculate_sprocket_travel(sprocket_pcd, motor_rpm, reduction_ratio)
                
                st.metric("초당 이동 거리", f"{results['per_sec_reduced']:.2f} mm/sec")
                st.write(f"스프라켓 원주: {results['circumference']:.2f} mm")
                st.write(f"분당 이동 거리: {results['per_min']:.2f} mm/min")
                st.write(f"초당 이동 거리: {results['per_sec']:.2f} mm/sec")
    
    elif menu == "PCD 계산기":
        st.header("⚡ PCD 계산기")
        tab1, tab2 = st.tabs(["자동기계용", "구동기어용"])
        
        with tab1:
            with st.form("pcd_calculator_auto"):
                col1, col2 = st.columns(2)
                pitch_inch = col1.selectbox("피치 (인치)", options=[6, 8, 10], index=1, key="auto_pitch_inch")
                teeth = col1.number_input("기어 톱니 수", min_value=1, value=21, step=1, key="auto_teeth")
                pcd = col2.number_input("PCD (mm)", min_value=0.1, value=1358.292, step=0.001, format="%.3f", key="auto_pcd")
                
                col1b, col2b = st.columns(2)
                calculate_pcd = col1b.form_submit_button("PCD 계산")
                calculate_teeth = col2b.form_submit_button("톱니 수 계산")
                
                if calculate_pcd:
                    calculated_pcd = calculate_pcd_vectorized(teeth, pitch_inch)
                    rounded_pcd = math.floor(calculated_pcd * 2) / 2
                    st.success(f"**계산된 PCD**: {rounded_pcd} mm")
                    
                if calculate_teeth:
                    calculated_teeth = (pcd * PI) / (pitch_inch * INCH_TO_MM)
                    st.success(f"**계산된 기어 톱니 수**: {calculated_teeth:.2f} 개")

        with tab2:
            with st.form("pcd_calculator_drive"):
                col1, col2 = st.columns(2)
                teeth_drive = col1.number_input("기어 톱니 수", min_value=1, value=20, step=1, key="drive_teeth")
                pcd_drive = col2.number_input("PCD (mm)", min_value=0.1, value=323.0, step=0.001, format="%.3f", key="drive_pcd")
                
                col1b, col2b = st.columns(2)
                calculate_pcd_drive = col1b.form_submit_button("PCD 계산")
                calculate_teeth_drive = col2b.form_submit_button("톱니 수 계산")
                
                if calculate_pcd_drive:
                    calculated_pcd_drive = (323 / 20) * teeth_drive
                    rounded_pcd_drive = int((calculated_pcd_drive + 4.9999) // 5 * 5)
                    st.success(f"**계산된 PCD**: {rounded_pcd_drive} mm")
                
                if calculate_teeth_drive:
                    calculated_teeth_drive = pcd_drive / (323 / 20)
                    st.success(f"**계산된 기어 톱니 수**: {calculated_teeth_drive:.2f} 개")
    
    elif menu == "Ollama 챗봇":
        asyncio.run(ollama_chat_async())

    elif menu == "예상견적가":
        # 비밀번호 인증 체크
        if not st.session_state.authenticated:
            with st.container():
                st.markdown("""
                <div style="background-color:#222; padding:25px; border-radius:15px; text-align:center;">
                    <h2 style='color:#39FF14;'>내부 견적 분석</h2>
                    <p style='font-size:1.2em;'>비밀번호를 입력하여 잠금 해제</p>
                </div>
                """, unsafe_allow_html=True)
                
                password = st.text_input("비밀번호", type="password", key="quote_password")
                
                if st.button("잠금 해제", use_container_width=True):
                    if password == PASSWORD:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("잘못된 비밀번호입니다.")
            return
        
        # 인증된 사용자에 대한 견적 화면
        pitch_type = st.session_state.pitch_type
        length = st.session_state.chain_length
        requirements = calculate_requirements(pitch_type, length)
        pitch_data = PITCH_TYPES.get(pitch_type, {'pitch': 6, 'density': 1})
        pitch_mm = pitch_data['pitch'] * INCH_TO_MM
        density = pitch_data['density']
        troly_per_m = (1000 / pitch_mm) * density
        troly_total = requirements['troly']

        # 수량 변경 알림
        if st.session_state.prev_troly is not None and troly_total > st.session_state.prev_troly:
            st.warning(f"트로리 수량 증가: {st.session_state.prev_troly} → {troly_total}조")
        st.session_state.prev_troly = troly_total

        # 상단 메트릭
        cols = st.columns(4)
        cols[0].metric("피치유형", pitch_type)
        cols[1].metric("피치 (mm)", f"{pitch_mm:.1f} mm")
        cols[2].metric("트로리/1m", f"{troly_per_m:.2f} 조")
        cols[3].metric("총 트로리", f"{troly_total} 조")

        st.caption(f"**계산식**: (1000 / ({pitch_data['pitch']} × 25.4)) × {density} × {length} = {troly_total}조")

        # 데이터 초기화
        df = init_data()
        for i, row in df.iterrows():
            part_id = row['규격']
            if part_id in requirements:
                df.at[i, '수량'] = requirements[part_id]

        # 샤클 단가 설정
        shackle_size = requirements.get('shackle_size', '6"')
        for shackle_id in ['shackle1', 'shackle2']:
            idx = df[df['규격'] == shackle_id].index
            if not idx.empty and shackle_size in SHACKLE_PRICES:
                price_info = SHACKLE_PRICES[shackle_size]
                df.at[idx[0], '구매 단가'] = price_info['purchase']
                df.at[idx[0], '판매 단가'] = price_info['sale']

        # 샤클 규격 선택
        with st.expander("샤클 규격 설정"):
            col1, col2 = st.columns(2)
            shackle1_size = col1.selectbox("1차 샤클 규격", SHACKLE_SIZES, index=SHACKLE_SIZES.index(shackle_size), key="shackle1_size")
            shackle2_size = col2.selectbox("2차 샤클 규격", SHACKLE_SIZES, index=SHACKLE_SIZES.index(shackle_size), key="shackle2_size")
            
            # 샤클 업데이트
            for shackle_id, size in [('shackle1', shackle1_size), ('shackle2', shackle2_size)]:
                idx = df[df['규격'] == shackle_id].index
                if not idx.empty and size in SHACKLE_PRICES:
                    price_info = SHACKLE_PRICES[size]
                    df.at[idx[0], '구매 단가'] = price_info['purchase']
                    df.at[idx[0], '판매 단가'] = price_info['sale']

        # 부품 테이블
        st.subheader("부품 목록 및 가격")
        
        # 품목 필터링
        base_items = ['메인 체인', '연결 링크', '트로리', '렌치볼트 (M6x35L)', '나이론 너트 (M6)', '트로리 조립 인건비']
        selectable_items = [item for item in df['품목'].tolist() if item not in base_items]
        
        if selectable_items:
            visible_items = st.multiselect(
                "추가 품목 선택",
                options=selectable_items,
                default=selectable_items
            )
            filtered_df = df[df['품목'].isin(base_items + visible_items)]
        else:
            filtered_df = df[df['품목'].isin(base_items)]

        # 데이터 에디터
        if st.session_state.quote_mode == "internal":
            edited_df = st.data_editor(filtered_df[['품목', '규격', '단위', '수량', '구매 단가', '판매 단가']])
        else:
            edited_df = st.data_editor(filtered_df[['품목', '규격', '단위', '수량', '판매 단가']])

        # 내부용 모드에서 이익률 조정
        if st.session_state.quote_mode == "internal":
            with st.expander("이익률 설정"):
                for part in PARTS:
                    filtered = edited_df[edited_df['규격'] == part['id']]
                    if not filtered.empty:
                        purchase_price = filtered.iloc[0]['구매 단가']
                        sale_price = filtered.iloc[0]['판매 단가']
                        margin = ((sale_price - purchase_price) / purchase_price * 100) if purchase_price > 0 else 0
                        
                        new_margin = st.slider(
                            f"{part['name']} 이익률 (%)", 
                            -100.0, 500.0, float(margin), step=1.0,
                            key=f"margin_{part['id']}"
                        )
                        new_sale_price = purchase_price * (1 + new_margin / 100)
                        edited_df.loc[edited_df['규격'] == part['id'], '판매 단가'] = new_sale_price

        # 최종 계산
        final_results = calculate_final_costs(edited_df, st.session_state.quote_mode)
        
        # 결과 표시
        st.subheader("최종 견적 요약")
        if st.session_state.quote_mode == "internal":
            cols = st.columns(4)
            cols[0].metric("총 구매 비용", f"{final_results['total_purchase']:,.0f} 원")
            cols[1].metric("총 판매 금액", f"{final_results['total_sale']:,.0f} 원")
            cols[2].metric("예상 이익", f"{final_results['total_profit']:,.0f} 원")
            cols[3].metric("이익률", f"{final_results['profit_margin']:.2f} %")
        else:
            cols = st.columns(2)
            cols[0].metric("총 견적 금액", f"{final_results['total_sale']:,.0f} 원")
            cols[1].metric("총 부품 수", f"{final_results['item_count']} 항목")

        # 테이블 표시
        if st.session_state.quote_mode == "internal":
            st.dataframe(edited_df[['품목', '단위', '수량', '구매 단가', '판매 단가', '구매 금액', '판매 금액']])
        else:
            st.dataframe(edited_df[['품목', '단위', '수량', '판매 단가', '판매 금액']])
        
        # 잠금 버튼
        if st.session_state.quote_mode == "internal":
            if st.button("🔒 잠금", use_container_width=True):
                st.session_state.authenticated = False
                st.rerun()

if __name__ == "__main__":
    main()

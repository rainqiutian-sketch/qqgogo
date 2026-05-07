import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
from datetime import datetime

# --- 설정 및 초기화 ---
st.set_page_config(page_title="고객 응대 AI 챗봇", layout="centered")
st.title("🛍️ 쇼핑몰 고객 만족 센터")

# 1. 환경 변수 또는 직접 입력을 통한 API 키 설정
if 'GEMINI_API_KEY' in st.secrets:
    api_key = st.secrets['GEMINI_API_KEY']
else:
    api_key = st.sidebar.text_input("Gemini API Key를 입력하세요", type="password")

# 2. 모델 선택 UI (사이드바)
selected_model_name = st.sidebar.selectbox(
    "사용할 모델 선택",
    ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash"], # 실제 지원되는 최신 모델명으로 조정
    index=0
)
st.sidebar.info(f"현재 모델: {selected_model_name}")

# 3. 대화 내역 초기화 및 상태 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 데이터 로드 및 시스템 프롬프트 구성 ---
def build_system_instruction():
    base_instruction = (
        "1. 당신은 쇼핑몰의 전문 고객 상담사입니다. 사용자의 불편/불만에 대해 정중하고 공감 어린 말투로 응답하세요.\n"
        "2. 사용자의 불편 사항을 구체적(무엇이/언제/어디서/어떻게)으로 정리하여 수집하고, 이를 사내 담당자에게 전달한다는 취지를 안내하세요.\n"
        "3. 대화 마지막 단계에서는 담당자의 회신을 위한 사용자 이메일 주소를 요청하세요. "
        "거부 시 '죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 직접 안내해 드리기 어렵습니다.'라고 정중히 마무리하세요.\n"
    )
    
    faq_path = "faq_data.csv"
    if os.path.exists(faq_path):
        try:
            df = pd.read_csv(faq_path)
            faq_markdown = df.to_markdown(index=False)
            # CSV 데이터가 있는 경우 추가 지침 삽입
            base_instruction += (
                f"\n[CSV 참조 데이터]\n{faq_markdown}\n\n"
                "4. 답변 시 위 참조 데이터를 우선 확인하세요. 데이터에 없는 내용은 임의로 지어내지 말고 "
                "'담당 부서 확인 후 안내해 드리겠습니다'라고 답변하세요."
            )
        except Exception as e:
            st.error(f"CSV 로드 중 오류 발생: {e}")
    
    return base_instruction

# --- 메인 챗봇 로직 ---
if api_key:
    genai.configure(api_key=api_key)
    
    # 모델 인스턴스 생성 (시스템 지침 포함)
    model = genai.GenerativeModel(
        model_name=selected_model_name,
        system_instruction=build_system_instruction()
    )

    # 대화 내역 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("불편하신 사항을 말씀해 주세요."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 토큰 최적화를 위해 최근 6턴(12개 메시지)만 컨텍스트로 유지
        history_context = st.session_state.messages[-12:]
        
        # Gemini 형식에 맞게 히스토리 변환 (user -> user, assistant -> model)
        formatted_history = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in history_context[:-1]
        ]

        with st.chat_message("assistant"):
            try:
                chat_session = model.start_chat(history=formatted_history)
                response = chat_session.send_message(prompt)
                
                full_response = response.text
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                # 429 ResourceExhausted 등 에러 처리
                error_msg = str(e)
                if "429" in error_msg or "ResourceExhausted" in error_msg:
                    st.error("현재 사용량이 많아 응답이 지연되고 있습니다. 1분 뒤에 다시 시도해 주세요.")
                else:
                    st.error(f"오류가 발생했습니다: {e}")

    # --- 유틸리티 기능 (사이드바 하단) ---
    st.sidebar.divider()
    
    # 1. 대화 초기화
    if st.sidebar.button("대화 초기화 (Clear)"):
        st.session_state.messages = []
        st.rerun()

    # 2. 대화 로그 다운로드 (CSV)
    if st.session_state.messages:
        log_df = pd.DataFrame(st.session_state.messages)
        csv_data = log_df.to_csv(index=False).encode('utf-8-sig')
        st.sidebar.download_button(
            label="전체 대화 내역 다운로드",
            data=csv_data,
            file_name=f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
else:
    st.warning("API 키를 입력하거나 st.secrets에 등록해야 서비스를 이용할 수 있습니다.")

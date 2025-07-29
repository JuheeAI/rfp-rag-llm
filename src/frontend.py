# src/frontend.py

import streamlit as st
import requests


##### UI #####
# 제목과 소제목 설정
st.title("RFP Analyzer")
st.caption("[ By 404AllFound ]")


##### 사이드바 #####
# 모델 선택을 위한 옵션
with st.sidebar:
    st.header("⚙️ Settings")
    
    model_source = st.selectbox(
        "Select Model",
        ("Open Source", "OpenAI"),
        key="model_select"
    )

    is_openai_selected = (model_source == "OpenAI")
    
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="",
        disabled=not is_openai_selected,
        key="api_key_input"
    )


##### 채팅 초기화 #####
# 새로고침 시 채팅 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []


##### 채팅 컨테이너 #####
# 채팅 기록 및 말풍선 형식으로 표시
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


#### 스트리밍 제네레이터 #####
# 아래 백엔드 주소로 모델 응답 API 요청
def stream_response_generator(payload: dict):
    # GCP VM 외부 IP:백엔드 포트
    API_URL = "http://34.63.229.25:9000"

    # API 정상 시 응답 데이터 처리
    try:
        with requests.post(f"{API_URL}/get_answer", json=payload, stream=True) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
    # API 비정상 시 에러 메시지 출력
    except requests.exceptions.RequestException as e:
        yield f"API Error: {e}"


##### 사용자 입력 및 응답 처리 #####
# 공백으로 채팅 입력창 생성
if prompt := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # OpenAI API 키 유효하지 않으면 에러 메시지 출력
    if is_openai_selected and not openai_api_key.startswith("sk-"):
        with chat_container:
            with st.chat_message("assistant"):
                error_msg = "Please enter a valid API key."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # OpenAI API 키 유효 & Open Source 모델 선택 시 백엔드로 API 요청
    else:
        with chat_container:
            with st.chat_message("assistant"):
                payload = {
                    "query": prompt,
                    "model_source": model_source,
                    "api_key": openai_api_key if is_openai_selected else None
                }
                response_content = st.write_stream(stream_response_generator(payload))
        
        # 채팅 기록
        st.session_state.messages.append({"role": "assistant", "content": response_content})

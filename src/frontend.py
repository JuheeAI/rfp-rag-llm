# src/frontend.py

import streamlit as st
import requests
import json


IP = "34.68.253.209"

########################################################################
# Notice 팝업 기능
@st.dialog("Notice")
def show_notice():
    st.markdown(
    """
    ### RFP?

    RFP(Request for Proposal)란 제안요청서의 약자로, 발주자가 특정 과제 수행에 필요한 요구사항을 정리한 문서입니다.

    ---
    ### RFP Analyzer 사용법

    (1) 사이드바에서 'Open Source' 또는 'OpenAI' 모델 선택

    (2) 'OpenAI' 모델을 선택했다면 유효한 API 키 입력 필요

    (3) 채팅창에 분석하고 싶은 RFP 관련 질문 입력

    ---
    ### 질문 예시

    - 고려대학교 차세대 포털·학사 정보시스템 구축 사업 제안서 요약해 줘. 예산도 함께 알려줘.
    > (정답) 사업예산 : 11,270,000,000원


    - 한영대학교 특성화 맞춤형 교육환경 구축 - 트랙운영 학사정보시스템 고도화 제안서 요약해 줘. 예산도 함께 알려줘.
    > (정답) 사업예산 : 130,000,000원

    """
    )
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("확인", use_container_width=True):
            st.session_state.notice_shown = True
            st.rerun()

if "notice_shown" not in st.session_state:
    show_notice()


########################################################################
# UI 설정
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("RFP Analyzer")
st.caption("[ By 404AllFound ]")

# 사이드바
with st.sidebar:
    st.header("⚙️ Settings")

    model_source = st.selectbox("Select Model", ("Open Source", "OpenAI"), key="model_select")
    is_openai_selected = (model_source == "OpenAI")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="", disabled=not is_openai_selected, key="api_key_input")

    st.divider()
    show_context = st.toggle("📚 Reference Check", value=True, key="show_context_toggle")
    st.divider()

    if st.button("Notice", use_container_width=True):
        show_notice()

    if st.button("Reset all chats", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


########################################################################
# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            content = message["content"]
            if st.session_state.show_context_toggle and isinstance(content, dict) and "context" in content:
                with st.expander("📚 Reference Check", expanded=False):
                    if content["context"]:
                        for doc in content["context"]:
                            source = doc.get('source', 'Unknown Source').replace('_', ' ').replace('.json', '')
                            doc_content = doc.get('content', '내용 없음')
                            st.write(f"📄 {source}")
                            st.info(doc_content)
                    else:
                        st.write("참조된 문서가 없습니다.")
                st.markdown(content.get("answer", "답변을 가져오지 못했습니다."))
            else:
                answer = content.get("answer") if isinstance(content, dict) else content
                st.markdown(answer)


########################################################################
# 사용자 입력 및 응답 처리
if prompt := st.chat_input(""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if is_openai_selected and not openai_api_key.startswith("sk"):
        with st.chat_message("assistant"):
            error_msg = "Please enter a valid API key."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        payload = {
            "query": prompt,
            "model_source": model_source,
            "api_key": openai_api_key if is_openai_selected else None
        }
        API_URL = f"http://{IP}:9000"

        try:
            with st.chat_message("assistant"):
                # nonlocal 대신 사용할 딕셔너리
                shared_data = {"context": None}

                def response_generator():
                    with requests.post(f"{API_URL}/get_answer", json=payload, stream=True) as response:
                        response.raise_for_status()
                        
                        buffer = ""
                        DELIMITER = "_|||_"
                        context_processed = False
                        
                        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                            if not context_processed:
                                buffer += chunk
                                if DELIMITER in buffer:
                                    context_json_str, llm_stream_part = buffer.split(DELIMITER, 1)
                                    # 딕셔너리에 컨텍스트 저장
                                    shared_data["context"] = json.loads(context_json_str)
                                    context_processed = True
                                    yield llm_stream_part
                            else:
                                yield chunk
                
                # 토글이 켜져 있을 때만 참조 문서 expander를 미리 생성
                if st.session_state.show_context_toggle:
                    context_expander = st.expander("📚 Reference Check", expanded=False)
                
                # 답변 스트리밍 시작
                answer_placeholder = st.empty()
                full_response = ""
                for chunk in response_generator():
                    full_response += chunk
                    answer_placeholder.markdown(full_response + "▌")
                answer_placeholder.markdown(full_response)
                
                # 스트리밍이 끝나고, 토글이 켜져 있으면 참조 문서 영역 채우기
                if st.session_state.show_context_toggle and shared_data["context"]:
                    with context_expander:
                        for doc in shared_data["context"]:
                            source = doc.get('source', 'Unknown Source').replace('_', ' ').replace('.json', '')
                            doc_content = doc.get('content', '내용 없음')
                            st.write(f"📄 {source}")
                            st.info(doc_content)

            # 전체 대화 내용을 세션 기록에 저장
            message_content = {
                "context": shared_data["context"],
                "answer": full_response
            }
            st.session_state.messages.append({"role": "assistant", "content": message_content})

        except requests.exceptions.RequestException as e:
            st.error(f"API Error: {e}")

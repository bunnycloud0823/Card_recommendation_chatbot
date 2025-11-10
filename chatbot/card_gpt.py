import streamlit as st
import json
import os
import re
import random
import datetime
from urllib.parse import quote
from dotenv import load_dotenv
from card_rag import search_card
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda
import gspread
from google.oauth2.service_account import Credentials


# ------------------------------- 초기 설정 -------------------------------
load_dotenv()
SHEET_ID = st.secrets["SHEET_ID"]

raw_json = st.secrets["GOOGLE_SERVICE_ACCOUNT"]
try:
    parsed = json.loads(raw_json)
    if isinstance(parsed, str):
        service_account_info = json.loads(parsed)
    else:
        service_account_info = parsed
except json.JSONDecodeError as e:
    st.error(f"JSON 파싱 오류: {e}")
    st.stop()

creds = Credentials.from_service_account_info(
    service_account_info,
    scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ],
)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SHEET_ID).sheet1


# ------------------------------- 로그 저장 함수 -------------------------------
def append_log_to_sheet(log_entry):
    try:
        row = [
            log_entry.get("timestamp"),
            log_entry.get("user_info", {}).get("name", ""),
            log_entry.get("user_info", {}).get("age_group", ""),
            log_entry.get("user_info", {}).get("occupation", ""),
            log_entry.get("query", ""),
            ", ".join(log_entry.get("card_ids", [])),
            ", ".join(log_entry.get("clicked_cards", [])),
            log_entry.get("session_duration_sec", 0),
            log_entry.get("ab_version", ""),
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        print(f"[로그 저장 실패] Google Sheets → {e}")


# ------------------------------- 세션 및 A/B 설정 -------------------------------
AB_VERSION = random.choice(["A", "B"])
SESSION_START = datetime.datetime.now()


# ------------------------------- 카드 링크·이미지 로드 -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LINK_IMAGE_PATH = os.path.join(BASE_DIR, "cards_link_image.json")

with open(LINK_IMAGE_PATH, "r", encoding="utf-8") as f:
    link_data = json.load(f)

LINK_DB = {str(item["card_id"]): item for item in link_data}


# ------------------------------- 카드 이름 추출 -------------------------------
def extract_card_name_by_id(text, card_id):
    """AI 응답에서 카드ID 앞의 줄 또는 문장을 추출"""
    pattern = rf"([\w가-힣A-Za-z\s]+)\s*\n?\s*카드ID\s*:\s*{card_id}"
    match = re.search(pattern, text)
    if match:
        name = match.group(1).strip()
        if "카드ID" in name:
            name = name.split("카드ID")[0].strip()
        return name

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "카드ID" in line and str(card_id) in line and i > 0:
            prev_line = lines[i - 1].strip()
            if prev_line:
                return prev_line
    return None


# ------------------------------- 카드 표시 -------------------------------
# user_info를 인수로 추가하여 카드별 신고 로직을 구현합니다.
def extract_card_ids(text):
    return re.findall(r"카드ID\s*:\s*(\d+)", text)


def make_naver_search_url(card_name: str) -> str:
    # 이 함수는 이미 urllib.parse.quote를 사용하여 URL 인코딩을 처리하고 있습니다.
    query = quote(card_name + " 카드 신청")
    return f"https://search.naver.com/search.naver?query={query}"


def show_card_details(card_ids, full_response_text=None, user_info=None):
    for cid in card_ids:
        data = LINK_DB.get(str(cid))
        if not data:
            continue

        card_name = data.get("card_name")
        if not card_name and full_response_text:
            card_name = extract_card_name_by_id(full_response_text, cid)

        # [문제 1 해결] 카드 이름이 없을 경우 카드 ID만 표시되는 문제를 해결
        if not card_name:
            # 기본적으로 ID만 남지 않도록 조금 더 명확한 문구를 사용
            card_name = f"카드 ({cid})"

        # 카드별 UI 컨테이너 및 신고 버튼 추가 (문제 2 및 3 해결)
        with st.container(border=True):
            st.markdown(f"**추천 카드: {card_name}**", unsafe_allow_html=True)

            img_path = data.get("image")
            if img_path:
                abs_img_path = os.path.normpath(
                    os.path.join(BASE_DIR, "..", img_path.replace("./", ""))
                )
                if os.path.exists(abs_img_path):
                    st.image(abs_img_path, width=250)

            pc_link = data.get("request_pc")
            m_link = data.get("request_m")

            if not pc_link and not m_link:
                apply_url = make_naver_search_url(card_name)
            else:
                apply_url = pc_link or m_link

            # 카드 신청 링크 표시
            st.markdown(
                f"[{card_name} 카드 신청 링크 열기]({apply_url})",
                unsafe_allow_html=True,
            )

            # [문제 2 및 3 해결] 카드별 오류 신고 버튼 및 로그 기록
            if user_info:
                # 고유 키 생성 (CID와 타임스탬프 결합)
                report_card_key = (
                    f"report_card_{cid}_{datetime.datetime.now().timestamp()}"
                )

                if st.button(f"🚨 '{card_name}' 정보 오류 신고", key=report_card_key):
                    report_log = {
                        "role": "system_log",
                        "content": (
                            f"사용자 '{user_info.get('name', '익명')}'이(가) 카드 ID {cid} ('{card_name}')의 정보 오류를 신고했습니다.\n"
                            f"신고 유형: 이미지/링크 오류. 신고된 카드 링크: {apply_url}"
                        ),
                    }
                    # 세션 메시지에 추가하여 로그 기록
                    st.session_state["messages"].append(report_log)
                    st.rerun()  # 로그가 즉시 반영되도록 Streamlit 다시 실행

        st.write("---")  # 카드 블록 구분선

    return ""


# ------------------------------- 세션 초기화 -------------------------------
if "pre_memory" not in st.session_state:
    st.session_state["pre_memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "안녕하세요. 저는 AI 카드 추천 전문가입니다. 당신에게 맞는 카드를 추천해드릴게요.",
        }
    ]

if "clicked_cards" not in st.session_state:
    st.session_state["clicked_cards"] = []


# ------------------------------- 모델 설정 -------------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

system_prompt = """
너는 카드사 직원이야. 고객의 질의가 들어오면 context에 따라 가장 혜택이 2개 추천해줘.
신용카드, 체크카드에 대한 명시가 없을 경우 신용카드, 체크카드 각각 1개씩 추천하고,
명시할 경우 해당 카드로 2개 추천해줘.
context 내용에 한해서만 추천해주되, context에 없는 내용은 발설하지 말아줘.
각 카드의 마지막 줄에는 반드시 '카드ID: {{card_id}}'를 포함시켜줘.

--출력 포맷--
해당란에 먼저 사용자가 어떤 카드를 원하는지 파악해서 요약본을 한 줄로 작성해줘.
추천카드명 
- 추천 이유 
- 해당 카드의 혜택
추천카드명 
- 추천 이유 
- 해당 카드의 혜택
"""

user_prompt = """
아래의 사용자 question을 읽고 context를 참고하여
가장 적합한 카드(사용자가 혜택을 최대로 받을 수 있는 카드)를 추천해주세요.

--chat_history--
{chat_history}

--question--
{question}

--context--
{context}
"""

final_prompt = ChatPromptTemplate([("system", system_prompt), ("user", user_prompt)])


def get_user_input(question):
    return {
        "chat_history": st.session_state["pre_memory"].chat_memory.messages,
        "question": question,
        "context": search_card(question),
    }


chain = RunnableLambda(get_user_input) | final_prompt | model | StrOutputParser()


# ------------------------------- 대화 함수 -------------------------------
def conversation_with_memory(question, user_info):
    stream_placeholder = st.empty()
    image_placeholder = st.empty()
    full_response = ""

    for chunk in chain.stream(question):
        full_response += chunk
        stream_placeholder.markdown(full_response)

    card_ids = extract_card_ids(full_response)

    with image_placeholder.container():
        # user_info를 show_card_details에 전달하여 신고 기능을 활성화
        show_card_details(card_ids, full_response, user_info)

    session_duration = (datetime.datetime.now() - SESSION_START).total_seconds()

    st.session_state["pre_memory"].save_context(
        {"input": question}, {"output": full_response}
    )

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_info": user_info,
        "query": question,
        "response": full_response,
        "card_ids": card_ids,
        "clicked_cards": st.session_state.get("clicked_cards", []),
        "session_duration_sec": session_duration,
        "ab_version": AB_VERSION,
    }

    append_log_to_sheet(log_entry)

    return full_response


# ------------------------------- 메인 화면 -------------------------------
st.title("AI의 맞춤 카드 추천 챗봇")

col1, col2 = st.columns(2)
with col1:
    age_group = st.radio(
        "연령대",
        ["10대", "20대", "30대", "40대", "50대 이상"],
        index=0,
    )

with col2:
    occupation = st.radio(
        "직업",
        ["학생", "직장인", "취업 준비생", "기타"],
        index=0,
    )

user_name = st.text_input("닉네임을 입력하세요:", "")

user_info = {
    "name": user_name or "익명",
    "age_group": age_group,
    "occupation": occupation,
}

# 기존 메시지 렌더링 및 'system_log' 처리
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        # 신고 로그는 일반 채팅과 구분되도록 경고 메시지로 표시합니다.
        if msg["role"] == "system_log":
            st.warning(msg["content"])
        else:
            st.markdown(msg["content"], unsafe_allow_html=True)

question = st.chat_input("메시지를 입력하세요. AI는 카드 추천만 가능해요.")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    if st.session_state["messages"][-1]["role"] != "assistant":
        # 응답과 버튼을 함께 관리하기 위해 컨테이너를 사용합니다.
        with st.container():
            try:
                # 1. AI 응답 생성 및 화면 렌더링
                # (ai_response와 show_card_details(카드별 버튼 포함)가 conversation_with_memory 내에서 모두 렌더링됩니다.)
                ai_response = conversation_with_memory(question, user_info)

                # 2. 세션 상태에 응답 추가 (로그 기록용)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": ai_response}
                )

                # [이전 신고 버튼 제거]: 카드별 신고 버튼은 show_card_details 내부에서 처리됩니다.

            except Exception as e:
                st.error(f"오류 발생: {e}")

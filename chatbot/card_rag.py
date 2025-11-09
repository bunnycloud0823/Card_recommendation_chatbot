# from langchain_community.document_loaders import JSONLoader
import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# 현재 파일 기준 절대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "cards_info.json"), "r", encoding="utf-8") as f:
    docs = json.load(f)


# vectorstore 객체 생성 함수 정의
def get_or_create_vectorstore(
    persist_directory="./Chroma", collection_name="card_info"
):
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore_exists = os.path.exists(persist_directory) and os.path.isdir(
        persist_directory
    )

    if vectorstore_exists:
        try:
            vectorstore = Chroma(
                embedding_function=embedding,
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
            if vectorstore._collection.count() > 0:
                print(
                    f"기존 vectorstore를 로드했습니다. (문서 수: {vectorstore._collection.count()})"
                )
                return vectorstore
            else:
                print("Vectorstore는 존재하지만 비어 있습니다. 새로 생성합니다.")
        except Exception as e:
            print(f"Vectorstore 로드 중 오류 발생: {e}")
            print("새로운 vectorstore를 생성합니다.")
    else:
        print("Vectorstore가 존재하지 않습니다. 새로 생성합니다.")

    # ⚠️ 존재하지 않을 경우 ⚠️
    # 새로운 vectorstore 생성
    cards_path = os.path.join(BASE_DIR, "cards_info.json")
    with open(cards_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    splitter = RecursiveCharacterTextSplitter()
    split_docs = splitter.create_documents([str(dict_) for dict_ in docs])

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    print(f"새로운 vectorstore를 생성했습니다. (문서 수: {len(split_docs)})")
    return vectorstore


def search_card(question, persist_directory="./Chroma", collection_name="card_info"):
    vectorstore = get_or_create_vectorstore(persist_directory, collection_name)
    retriever = vectorstore.as_retriever()
    result = retriever.invoke(question)

    card_context = []
    for page in result:
        card_context.append(page.page_content)

    return card_context

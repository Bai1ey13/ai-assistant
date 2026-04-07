import streamlit as st
import os
import time
import json
import re
import requests
import pytesseract
from PIL import Image
from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 指定 tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'E:\tesseract-ocr\tesseract.exe'

# ==================== 页面配置 ====================
st.set_page_config(page_title="计算机网络AI助教", page_icon="🌐")

# ==================== 配置 ====================
API_KEY = "sk-6b22072a630f44648a4bd89525de5887"
client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# ==================== 教材知识库 ====================
@st.cache_resource
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return db

db = load_knowledge_base()

def retrieve_context(query, k=4):
    docs = db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

# ==================== 教材上传与管理 ====================
def extract_text_from_pdf(file):
    from pypdf import PdfReader
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    from docx import Document
    doc = Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def update_knowledge_base(uploaded_files):
    lib_dir = "./教材_库"
    os.makedirs(lib_dir, exist_ok=True)
    for uploaded_file in uploaded_files:
        base_name = uploaded_file.name
        save_path = os.path.join(lib_dir, base_name)
        if os.path.exists(save_path):
            name, ext = os.path.splitext(base_name)
            save_path = os.path.join(lib_dir, f"{name}_{int(time.time())}{ext}")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    all_text = ""
    for file_name in os.listdir(lib_dir):
        file_path = os.path.join(lib_dir, file_name)
        try:
            if file_name.lower().endswith(".pdf"):
                with open(file_path, "rb") as f:
                    text = extract_text_from_pdf(f)
            elif file_name.lower().endswith(".docx"):
                with open(file_path, "rb") as f:
                    text = extract_text_from_docx(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            all_text += text + "\n\n"
        except Exception as e:
            print(f"处理文件 {file_name} 失败：{e}")
            continue
    if not all_text:
        return 0, 0
    temp_txt = "./教材/merged.txt"
    os.makedirs("./教材", exist_ok=True)
    with open(temp_txt, "w", encoding="utf-8") as f:
        f.write(all_text)
    loader = TextLoader(temp_txt, encoding="utf-8")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")
    db_new = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    db_new.persist()
    chunk_count = len(texts)
    file_count = len(os.listdir(lib_dir))
    return chunk_count, file_count

def list_knowledge_files():
    lib_dir = "./教材_库"
    if not os.path.exists(lib_dir):
        return []
    return [f for f in os.listdir(lib_dir) if os.path.isfile(os.path.join(lib_dir, f))]

def delete_knowledge_file(filename):
    file_path = os.path.join("./教材_库", filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

# ==================== AI 问答核心函数 ====================
def is_meta_question(query):
    meta_keywords = ["你的功能", "你是谁", "什么模式", "你能做什么", "你是什么", "模式是什么", "当前模式"]
    for kw in meta_keywords:
        if kw in query:
            return True
    return False

def meta_answer(query, mode):
    if "你的功能" in query or "你能做什么" in query or "你的作用" in query:
        return f"""我是《计算机网络》课程的AI助教，我可以：
- 解答教材中的知识点
- 指导实验操作步骤
- 提供习题练习

当前模式为：**{mode}**。你可以通过侧边栏切换模式。"""
    elif "什么模式" in query or "当前模式" in query or "模式是什么" in query:
        return f"当前是 **{mode}**。你可以通过侧边栏切换模式。"
    else:
        return f"我是你的《计算机网络》AI助教，当前模式为：**{mode}**。有什么关于教材的问题吗？"

def is_context_sufficient(query, context):
    prompt = f"""判断以下教材内容是否足够回答用户问题。只回答“足够”或“不足够”。

教材内容：
{context[:800]}

用户问题：{query}

回答："""
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0
    )
    result = response.choices[0].message.content.strip()
    return "足够" in result

def rewrite_query(query):
    prompt = f"""请将以下用户问题改写成更具体、更关键词化的检索查询。只输出改写后的查询，不要解释。

原始问题：{query}

改写后的查询："""
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def deep_retrieve(query, max_rounds=2):
    context = retrieve_context(query, k=4)
    for _ in range(max_rounds - 1):
        if is_context_sufficient(query, context):
            break
        new_query = rewrite_query(query)
        new_context = retrieve_context(new_query, k=4)
        context += "\n\n" + new_context
    if len(context) > 2500:
        context = context[:2500]
    return context

def get_ai_answer(user_message: str, mode: str = "教材检索（一般）", context: str = None) -> str:
    if is_meta_question(user_message):
        return meta_answer(user_message, mode)
    if mode == "通用模式":
        system_prompt = f"""你是计算机网络助教。请用通俗易懂、内容丰富的语言回答用户的问题，不基于特定教材。可以适当举例。

用户问题：{user_message}"""
    else:
        if context is None:
            if mode == "教材检索（一般）":
                context = retrieve_context(user_message, k=4)
            else:
                context = deep_retrieve(user_message)
        if len(context) > 2500:
            context = context[:2500]
        system_prompt = f"""你是计算机网络助教。请基于以下教材内容回答用户的问题。要求：
- 用自己的话概括，不要逐字复述。
- 回答要详细、完整，尽可能解释清楚原理，不要过度压缩字数。
- 如果教材内容与问题完全不相关，请明确说“教材中没有找到相关信息”，然后可以用你的通用知识简要补充。

教材内容：
{context}

用户问题：{user_message}

回答："""
    response = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "system", "content": system_prompt}]
    )
    return response.choices[0].message.content

# ==================== Streamlit 界面 ====================
st.title("📚 中职《计算机网络》AI助教")
st.markdown("---")

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "教材检索（一般）"
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "file_count" not in st.session_state:
    st.session_state.file_count = 0
if "last_update" not in st.session_state:
    st.session_state.last_update = "从未"

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("🤖 关于本助教")
    st.markdown("支持教材问答与教材管理。")
    st.markdown("---")

    # 模式选择（明确放在侧边栏）
    st.subheader("⚙️ 模式选择")
    selected_mode = st.radio(
        "回答模式",
        ["通用模式", "教材检索（一般）", "教材检索（深度）"],
        index=1,
        help="通用模式：不基于教材，自由对话；一般检索：单次RAG；深度检索：多轮RAG（Agentic）"
    )
    st.session_state.mode = selected_mode
    st.markdown("---")

    # ocr
    st.subheader("🖼️ 图片文字提取")
    uploaded_img = st.file_uploader("上传教材截图或拓扑图", type=["png", "jpg", "jpeg"], key="ocr_upload")
    if uploaded_img:
        with st.spinner("正在识别图片文字..."):
            img = Image.open(uploaded_img)
            # 使用中文+英文语言包
            ocr_text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            st.session_state.ocr_text = ocr_text
            st.success(f"识别到 {len(ocr_text)} 个字符，已加入上下文。")
            with st.expander("查看识别结果"):
                st.write(ocr_text)
    else:
        # 可选：如果不想保留上次的OCR结果，可以清空
        if "ocr_text" in st.session_state:
            st.session_state.ocr_text = ""

    # 教材管理
    st.subheader("📚 教材管理")
    uploaded_books = st.file_uploader(
        "上传教材文件（支持 PDF / Word / TXT）",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="book_upload"
    )
    if uploaded_books:
        if st.button("更新知识库"):
            with st.spinner("正在处理教材，请稍候..."):
                chunks, files = update_knowledge_base(uploaded_books)
                if chunks:
                    st.session_state.chunk_count = chunks
                    st.session_state.file_count = files
                    st.session_state.last_update = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.success(f"知识库已更新，共 {chunks} 个文本块。")
                    st.cache_resource.clear()
                else:
                    st.error("更新失败，请检查文件内容。")
    st.markdown("---")
    st.subheader("📖 已上传教材")
    knowledge_files = list_knowledge_files()
    if not knowledge_files:
        st.info("暂无教材，请上传。")
    else:
        for file in knowledge_files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(file)
            with col2:
                if st.button("删除", key=f"del_{file}"):
                    if delete_knowledge_file(file):
                        with st.spinner("正在重建知识库..."):
                            chunks, files = update_knowledge_base([])
                            if chunks:
                                st.session_state.chunk_count = chunks
                                st.session_state.file_count = files
                                st.session_state.last_update = time.strftime("%Y-%m-%d %H:%M:%S")
                                st.success(f"已删除 {file}，知识库已更新。")
                                st.cache_resource.clear()
                            else:
                                st.error("重建失败。")
                        st.rerun()
    st.markdown("---")
    st.subheader("📊 知识库状态")
    st.markdown(f"- **文件数**：{st.session_state.file_count}")
    st.markdown(f"- **文本块数**：{st.session_state.chunk_count}")
    st.markdown(f"- **最后更新**：{st.session_state.last_update}")

# ==================== 主聊天区域 ====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    current_mode = st.session_state.mode

    if current_mode == "通用模式":
        with st.chat_message("assistant"):
            with st.spinner("AI 思考中..."):
                answer = get_ai_answer(prompt, mode=current_mode)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        with st.spinner("正在检索知识库..."):
            if current_mode == "教材检索（一般）":
                context = retrieve_context(prompt, k=4)
            else:
                context = deep_retrieve(prompt)
            # 如果有 OCR 文本，追加到上下文
            if "ocr_text" in st.session_state and st.session_state.ocr_text:
                context += "\n\n[用户上传的图片中的文字]：\n" + st.session_state.ocr_text
            with st.expander("📖 查看检索到的教材内容（共 " + str(len(context)) + " 字符）"):
                st.markdown(context)
        with st.chat_message("assistant"):
            with st.spinner("AI 助教正在思考..."):
                answer = get_ai_answer(prompt, mode=current_mode, context=context)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
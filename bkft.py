import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 教材文件夹路径
BOOKS_PATH = "./教材"

# 加载所有txt文件
documents = []
for file in os.listdir(BOOKS_PATH):
    if file.endswith(".txt"):
        file_path = os.path.join(BOOKS_PATH, file)
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
        print(f"已加载：{file}")

print(f"共加载 {len(documents)} 个文档片段")

# 文本分割（每块500字符，重叠50）
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"分割后得到 {len(texts)} 个文本块")

# 使用中文嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

# 创建向量数据库并持久化
db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
db.persist()
print("知识库构建完成，已保存到 ./chroma_db 文件夹")
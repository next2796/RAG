from langchain_community.vectorstores import FAISS # 导入FAISS向量存储库
from langchain_huggingface import HuggingFaceEmbeddings # 导入Hugging Face嵌入模型
from langchain_community.document_loaders import TextLoader # 导入文本加载器
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 导入递归字符文本分割器
from langchain_openai import ChatOpenAI # 导入ChatOpenAI模型
# 使用 OpenAI API 的 ChatOpenAI 模型
chat_model = ChatOpenAI(
    openai_api_key="ollama", # ollama兼容OpenAI API的格式
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b"
)
# 加载文本文件 "sanguoyanyi.txt"，编码格式为 'utf-8'
loader = TextLoader("E:\shi yan dai ma er\RAG2\sanguoyanyi.txt", encoding='utf-8')
docs = loader.load() # 将文件内容加载到变量 docs 中
# 把文本分割成 200 字一组的切片，每组之间有 20 字重叠
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200,
chunk_overlap=20)
chunks = text_splitter.split_documents(docs) # 将文档分割成多个小块
# print(chunks)
# 初始化嵌入模型，使用预训练的语言模型 'bge-large-zh-v1___5'
embedding = HuggingFaceEmbeddings(model_name='models/AI-ModelScope/bge-large-zh-v1___5')
# 构建 FAISS 向量存储和对应的 retriever
vs = FAISS.from_documents(chunks, embedding) # 将文本块转换为向量并存储在FAISS中
retriever = vs.as_retriever() # 创建一个检索器用于从向量存储中获取相关信息
from langchain.chains import RetrievalQA # 导入RetrievalQA链
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
# 创建一个系统消息，用于定义机器人的角色
system_message = SystemMessagePromptTemplate.from_template(
    "根据以下已知信息回答用户问题。\n 已知信息{context}"
)
# 创建一个人类消息，用于接收用户的输入
human_message = HumanMessagePromptTemplate.from_template(
    "用户问题：{question}"
)
# 将这些模板结合成一个完整的聊天提示
chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message,
])
# 定义链的类型参数，包括使用的提示模板
chain_type_kwargs = {"prompt": chat_prompt}
# 创建一个问答链，将语言模型、检索器和提示模板结合起来
# chat_model:生成回答的语言模型, stuff:所有检索到的文档内容合并成一个大文本块，然后传递给语言模型。
# retriever: 之前创建的一个 FAISS 检索器实例。它的作用是从 FAISS 向量存储中找到与用户问题最相关的文档或文本块。这些相关的文档会被传递给语言模型以生成回答。
# chain_type_kwargs 是一个字典，包含了用于配置问答链的一些关键参数。
qa = RetrievalQA.from_chain_type(llm=chat_model,
chain_type="stuff", retriever=retriever,
chain_type_kwargs=chain_type_kwargs)
# 用户的问题
user_question = "五虎上将有哪些？"
# 使用检索器获取与问题相关的文档
related_docs = retriever.invoke(user_question)
# 打印相关文档的内容
print("相关文档:")
for i, doc in enumerate(related_docs):
    print(f"文档 {i+1}:")
    print(doc.page_content)
    print("-" * 40)
# 使用问答链来回答问题 "五虎上将有哪些？" 并打印结果
print(qa.invoke(user_questioon))

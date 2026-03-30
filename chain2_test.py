# 使用LCEL表达式构造自定义链
from langchain_core.output_parsers import StrOutputParser # 导入字符串输出解析器
from langchain_core.prompts import ChatPromptTemplate # 导入聊天提示模板
from langchain_openai import ChatOpenAI # 导入ChatOpenAI模型
# 使用 OpenAI API 的 ChatOpenAI 模型
chat_model = ChatOpenAI(
    openai_api_key="ollama", # ollama兼容OpenAI API的格式
    base_url="http://localhost:11434/v1",
    model="qwen2.5_0.5b"    
)
# 创建一个聊天提示模板，其中包含占位符 {topic}
prompt = ChatPromptTemplate.from_template("说出一句包含{topic}的诗句。")
# 创建一个字符串输出解析器，用于将语言模型的输出解析为字符串
output_parser = StrOutputParser()
# 构造一个链，依次包含提示模板、语言模型和输出解析器
chain = prompt | chat_model | output_parser
# 使用链来生成回答，并打印结果
print(chain.invoke({"topic": "花"}))
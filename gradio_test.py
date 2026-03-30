import gradio as gr
import requests
# 定义后台的fastapi的URL
backend_url = "http://127.0.0.1:6066/chat"
def chat_with_backend(prompt, history, sys_prompt, history_len, temperature, top_p,
max_tokens, stream):
    # history:["role": "user", "metadata":{'title':None},"content":"xxxx"],去掉metadata字段
    history_none_meatdata = [{"role": h.get("role"), "content": h.get("content")} for h in
history]
    # print(history)
    # 构建请求的数据
    data = {
        "query": prompt,
        "sys_prompt": sys_prompt,
        "history": history_none_meatdata,   
        "history_len": history_len,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    response = requests.post(backend_url, json=data, stream=True)
    if response.status_code == 200:
        chunks = ""
        if stream:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
                yield chunks
        else:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
            yield chunks
# 使用gr.Blocks创建一个块，并设置可以填充高度和宽度
with gr.Blocks(fill_width=True, fill_height=True) as demo:
    # 创建一个标签页
    with gr.Tab("🤖 聊天机器人"):
        # 添加标题
        gr.Markdown("## 🤖 聊天机器人") 
        # 创建一个行布局
        with gr.Row():
            # 创一个左侧的列布局
            with gr.Column(scale=1, variant="panel") as sidebar_left:
                sys_prompt = gr.Textbox(label="系统提示词", value="You are a helpful assistant")
                history_len = gr.Slider(minimum=1, maximum=10, value=1, label="保留历史对话的数量")
                temperature = gr.Slider(minimum=0.01, maximum=2.0, value=0.5, step=0.01,label="temperature")
                top_p = gr.Slider(minimum=0.01, maximum=1.0, value=0.5, step=0.01,label="top_p")
                max_tokens = gr.Slider(minimum=512, maximum=4096, value=1024, step=8,label="max_tokens")
                stream = gr.Checkbox(label="stream", value=True)
            # 创建右侧的列布局，设置比例为10
            with gr.Column(scale=10) as main:
                # 创建聊天机器人的聊天界面，高度为500px
                chatbot = gr.Chatbot(type="messages", height=500)
                # 创建chatinterface, 用于处理聊天的逻辑
                gr.ChatInterface(fn=chat_with_backend,
                                type="messages",
                                chatbot=chatbot,
                                additional_inputs=[
                                sys_prompt,
                                history_len,
                                temperature,
                                top_p,
                                max_tokens, 
                                stream
                            ])
demo.launch()
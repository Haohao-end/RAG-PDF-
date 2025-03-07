import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
import ollama
import re


def process_pdf(pdf_bytes):
    """
    处理PDF文件并创建向量存储
    Args:
        pdf_bytes: PDF文件的路径
    Returns:
        tuple: 文本分割器、向量存储和检索器
    """
    if pdf_bytes is None:
        return None, None, None
    # 加载PDF文件
    loader = PyMuPDFLoader(pdf_bytes)
    data = loader.load()
    # 创建文本分割器，设置块大小为500，重叠为100
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    # 使用Ollama的deepseek-r1模型创建嵌入
    embeddings = OllamaEmbeddings(model="deepseek-r1:8b")
    # 将Chroma替换为FAISS向量存储
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    # 从向量存储中创建检索器
    retriever = vectorstore.as_retriever()
    #  # 返回文本分割器、向量存储和检索器
    return text_splitter, vectorstore, retriever


def combine_docs(docs):
    """
    将多个文档合并为单个字符串
    Args:
        docs: 文档列表
    Returns:
        str: 合并后的文本内容
    """
    return "\n\n".join(doc.page_content for doc in docs)


def ollama_llm(question, context, chat_history):
    """
    使用Ollama模型生成回答
    Args:
        question: 用户问题
        context: 相关上下文
        chat_history: 聊天历史记录
    Returns:
        str: 模型生成的回答
    """
    # 构建更清晰的系统提示和用户提示
    system_prompt = """你是一个专业的AI助手。请基于提供的上下文回答问题。
    - 回答要简洁明了，避免重复
    - 如果上下文中没有相关信息，请直接说明
    - 保持回答的连贯性和逻辑性"""
    
    # 只保留最近的3轮对话历史，避免上下文过长
    recent_history = chat_history[-3:] if len(chat_history) > 3 else chat_history
    chat_history_text = "\n".join([f"Human: {h}\nAssistant: {a}" for h, a in recent_history])
    
    # 构建更结构化的提示模板
    user_prompt = f"""基于以下信息回答问题：
                    问题：{question}
                    相关上下文：
                    {context}
                    请用中文回答上述问题。回答要简洁准确，避免重复。"""
    
    # 调用Ollama模型生成回答
    response = ollama.chat(
        model="deepseek-r1:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    response_content = response["message"]["content"]
    # 移除思考过程和可能的重复内容
    final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    
    return final_answer


def rag_chain(question, text_splitter, vectorstore, retriever, chat_history):
    """
    实现RAG（检索增强生成）链
    Args:
        question: 用户问题
        text_splitter: 文本分割器
        vectorstore: 向量存储
        retriever: 检索器
        chat_history: 聊天历史
    Returns:
        str: 生成的回答
    """
    # 减少检索文档数量，提高相关性
    retrieved_docs = retriever.invoke(question, {"k": 2})
    
    # 优化文档合并方式，去除可能的重复内容
    formatted_content = "\n".join(set(doc.page_content.strip() for doc in retrieved_docs))
    
    return ollama_llm(question, formatted_content, chat_history)


def chat_interface(message, history, pdf_bytes=None, text_splitter=None, vectorstore=None, retriever=None):
    """
    聊天接口函数，处理用户输入并返回回答
    Args:
        message: 用户消息
        history: 聊天历史
        pdf_bytes: PDF文件
        text_splitter: 文本分割器
        vectorstore: 向量存储
        retriever: 检索器
    Returns:
        str: 生成的回答
    """
    if pdf_bytes is None:
        # 无PDF文件的普通对话模式
        response = ollama_llm(message, "", history)
    else:
        # 有PDF文件的RAG对话模式
        response = rag_chain(message, text_splitter, vectorstore, retriever, history)
    return response


def create_chat_interface():
    """
    创建Gradio聊天界面

    Returns:
        gr.Blocks: Gradio界面对象
    """
    # 创建一个用户界面，并应用了一些自定义的CSS样式。
    with gr.Blocks() as demo:
        # 定义状态变量用于存储PDF处理相关的对象
        pdf_state = gr.State(None)  
        # 存储文本分割器对象,用于将PDF文本分割成小块
        text_splitter_state = gr.State(None) 
        # 存储向量数据库对象,用于存储文本向量
        vectorstore_state = gr.State(None)  
        # 存储检索器对象,用于检索相关文本片段
        retriever_state = gr.State(None)  

        with gr.Column(elem_classes="container"):
            # 创建界面组件
            with gr.Column(elem_classes="header"):
                gr.Markdown("# PDF智能问答助手")
                gr.Markdown("上传PDF文档，开始智能对话")

            # 文件上传区域
            with gr.Column(elem_classes="file-upload"):
                file_output = gr.File(
                    label="上传PDF文件",
                    file_types=[".pdf"],
                    file_count="single"
                )
                
                # 处理PDF上传
                def on_pdf_upload(file):
                    """
                    处理PDF文件上传
                    
                    Args:
                        file: 上传的文件对象
                        
                    Returns:
                        tuple: 包含处理后的PDF相关对象
                    """
                    # 如果文件存在
                    if file is not None:
                        # 处理PDF文件,获取文本分割器、向量存储和检索器
                        text_splitter, vectorstore, retriever = process_pdf(file.name)
                        # 返回文件对象和处理后的组件
                        return file, text_splitter, vectorstore, retriever
                    # 如果文件不存在,返回None值
                    return None, None, None, None
                
                # 注册文件上传事件处理
                file_output.upload(
                    # 当文件上传时调用on_pdf_upload函数处理
                    on_pdf_upload, 
                    # inputs参数指定输入组件为file_output
                    inputs=[file_output],
                    # outputs参数指定输出状态变量
                    outputs=[pdf_state, text_splitter_state, vectorstore_state, retriever_state]
                )

            # 聊天区域
            with gr.Column(elem_classes="chat-container"):
                chatbot = gr.Chatbot(
                    height=500,
                    bubble_full_width=False,
                    show_label=False,
                    avatar_images=None,
                    elem_classes="chatbot"
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="输入问题",
                        placeholder="请输入你的问题...",
                        scale=12,
                        container=False
                    )
                    send_btn = gr.Button("发送", scale=1, variant="primary")

                with gr.Row(elem_classes="button-row"):
                    clear = gr.Button("清空对话", variant="secondary")
                    regenerate = gr.Button("重新生成", variant="secondary")

        # 发送消息处理函数
        def respond(message, chat_history, pdf_bytes, text_splitter, vectorstore, retriever):
            """
            处理用户消息并生成回答
            
            Args:
                message: 用户消息
                chat_history: 聊天历史
                pdf_bytes: PDF文件
                text_splitter: 文本分割器
                vectorstore: 向量存储
                retriever: 检索器
                
            Returns:
                tuple: (清空的消息框, 更新后的聊天历史)
            """
            # 如果用户消息为空(去除首尾空格后),直接返回空消息和原聊天历史
            if not message.strip():
                return "", chat_history
                
            # 调用chat_interface函数处理用户消息,生成回复
            bot_message = chat_interface(
                message, 
                chat_history, 
                pdf_bytes, 
                text_splitter, 
                vectorstore, 
                retriever
            )
            
            # 将用户消息和模型回复作为一轮对话添加到聊天历史中
            chat_history.append((message, bot_message))
            
            # 返回空消息(清空输入框)和更新后的聊天历史
            return "", chat_history

        # 事件处理
        # 当用户按回车键提交消息时触发
        msg.submit(
            respond,
            [msg, chatbot, pdf_state, text_splitter_state, vectorstore_state, retriever_state],
            [msg, chatbot]
        )
        
        # 当用户点击发送按钮时触发
        send_btn.click(
            respond,
            [msg, chatbot, pdf_state, text_splitter_state, vectorstore_state, retriever_state],
            [msg, chatbot]
        )
        
        # 当用户点击清空按钮时触发
        # lambda: (None, None) 返回两个None值来清空消息框和对话历史
        # queue=False 表示不进入队列直接执行
        clear.click(lambda: (None, None), None, [msg, chatbot], queue=False)
        
        # 重新生成按钮功能
        def regenerate_response(chat_history, pdf_bytes, text_splitter, vectorstore, retriever):
            """
            重新生成最后一条回答
            
            Args:
                chat_history: 聊天历史
                pdf_bytes: PDF文件
                text_splitter: 文本分割器
                vectorstore: 向量存储
                retriever: 检索器
                
            Returns:
                list: 更新后的聊天历史
            """
            # 如果聊天历史为空,直接返回
            if not chat_history:
                return chat_history
                
            # 获取最后一条用户消息
            last_user_message = chat_history[-1][0]
            
            # 移除最后一轮对话
            chat_history = chat_history[:-1]  
            
            # 使用chat_interface重新生成回答
            bot_message = chat_interface(
                last_user_message,  # 最后一条用户消息
                chat_history,       # 更新后的聊天历史
                pdf_bytes,          # PDF文件内容
                text_splitter,      # 文本分割器
                vectorstore,        # 向量存储
                retriever          # 检索器
            )
            
            # 将新生成的对话添加到历史中
            chat_history.append((last_user_message, bot_message))
            
            # 返回更新后的聊天历史
            return chat_history

        # 为重新生成按钮绑定点击事件
        # 当点击时调用regenerate_response函数
        # 输入参数为chatbot等状态
        # 输出更新chatbot显示
        regenerate.click(
            regenerate_response,
            [chatbot, pdf_state, text_splitter_state, vectorstore_state, retriever_state],
            [chatbot]
        )

    return demo


# 启动接口
if __name__ == "__main__":
    """
    主程序入口：启动Gradio界面
    """
    demo = create_chat_interface()
    demo.launch(
        server_name="127.0.0.1", 
        server_port=8888,
        show_api=False,
        share=False
    )



from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnableLambda , RunnablePassthrough
from langchain_core.messages import SystemMessage , HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI,HarmCategory, HarmBlockThreshold
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from base64 import b64decode
import embedding as emb
import pickle #准备解冻

import streamlit as st #webdevelopment
from PIL import Image
import io
import base64

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.globals import set_debug

# 开启全局调试模式
set_debug(True)


st.set_page_config(page_title = 'I love Kerian', layout= 'wide')
st.title('Agentic Multimodal Rag System for Attention is all your need')

def is_valid_image(data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data))
        return True
    except:
        return False

@st.cache_resource #只运行一次
def load_agent() :
    def parse_docs(docs):
        b64_images = []
        texts = []

        for doc in docs:
            content = doc.page_content if hasattr(doc, 'page_content') else doc

            # 针对 Bytes 类型的处理 (Pickle文本/表格 或 Base64图片)
            if isinstance(content, bytes):
                # 1. 尝试解冻文本或表格 (对应 pickle.dumps)
                try:
                    unwrapped_doc = pickle.loads(content)
                    texts.append(unwrapped_doc)
                    continue
                except Exception:
                    pass

                # 2. 如果解冻失败，说明这是你 encode 进来的 Base64 字符串！
                try:
                    # 把 bytes 重新转换回字符串
                    content_str = content.decode('utf-8')
                    # 判断特征头
                    if "iVB0" in content_str[:50] or "/9j/" in content_str[:50]:
                        b64_images.append(content_str)
                        continue
                except Exception:
                    pass

            # 针对纯 String 类型的处理 (以防万一)
            elif isinstance(content, str):
                if "iVB0" in content[:50] or "/9j/" in content[:50]:
                    b64_images.append(content)
                else:
                    texts.append(content)

        return {'images': b64_images, 'texts': texts}

    @tool
    def search_pdf_database(query: str) -> list:
        """
        检索本地《Attention Is All You Need》论文数据库。
        🚨【重要搜索规范】：
        1. 如果用户问题涉及架构图、流程图或指定了图片，你的 query 必须强制包含 "Figure"、"Image" 或 "architecture" 等视觉关键词。
        2. 如果用户问题涉及精准数据查询、BLEU分数、成本等，你的 query 必须强制包含 "Table" 关键词！
        """
        # 👇 核心修复点：强行破解 LangChain 默认只搜 4 条的限制！把它扩大到 15 条！
        if hasattr(emb.retriever, 'search_kwargs'):
            emb.retriever.search_kwargs['k'] = 15
        else:
            emb.retriever.search_kwargs = {'k': 15}
            
        docs = emb.retriever.invoke(query) 
        
        # 👇 把这里的 8 改成 15，给排在后面的图片留出进入大模型嘴里的空间！
        docs = docs[:15] 
        parsed_data = parse_docs(docs)
        
        content_blocks = []
        text_list = []
        
        # ... 下面的 html 表格和 base64 拼装代码保持你现在的样子，不用动 ...
        
        # 拼装文字和表格（完美保留 HTML 结构！）
        if len(parsed_data['texts']) > 0:
            for element in parsed_data['texts']:
                element_type = str(type(element)).lower()
                
                # 修复表格 Bug：强制提取 text_as_html，保留完美的二维行列结构！
                if 'table' in element_type:
                    if hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html') and element.metadata.text_as_html:
                        html_table = element.metadata.text_as_html
                        text_list.append(f"\n--- [表格数据开始 (HTML格式)] ---\n{html_table}\n--- [表格数据结束] ---\n")
                    else:
                        text_list.append(f"\n--- [表格数据] ---\n{element.text}\n")
                else:
                    text_val = element.text if hasattr(element, 'text') else str(element)
                    text_list.append(text_val)
                    
        context_text = '\n\n'.join(text_list)
        if not context_text:
            context_text = "没有找到相关的文字或表格资料。"
            
        content_blocks.append({
            "type": "text", 
            "text": f"检索到的文字与表格信息如下：\n{context_text}"
        })
        
        # 拼装图片
        if len(parsed_data['images']) > 0:
            for b64_img in parsed_data['images']:
                prefix = "" if b64_img.startswith("data:image") else "data:image/jpeg;base64,"
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"{prefix}{b64_img}"}
                })
                
        print(f"\n[Tool日志] 数据库搜出了 {len(parsed_data['texts'])} 段文字/表格, 成功提取 {len(parsed_data['images'])} 张图片！\n")
                
        return content_blocks

    # --- 3. 工具 B：Tavily 联网搜索工具 ---
    search_specific_websites = TavilySearchResults(
        max_results=3, 
        search_depth="advanced", 
        include_domains=[
            "lilianweng.github.io", 
            "wikipedia.org"
        ],
        description="当用户询问最新的知识，或者本地 PDF 数据库里查不到的信息时，使用这个工具去特定的外部网站搜索。" 
    )

    # --- 4. 组装工具箱 ---
    tools = [search_pdf_database, search_specific_websites]

    # --- 5. 准备大模型 ---
    # 只有原生支持多模态的模型（如 gemini-2.5-flash）才能吃得下上面的图片积木
    llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview',temperature = 0 ,safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)

    # --- 6. 召唤终极 Agent ---
    agent = create_agent(
        model=llm, 
        tools=tools, 
        system_prompt=(
            "你是一个极其聪明、支持视觉处理的学术与网络助手。你现在有两个工具：\n"
            "1. 如果问题是关于你本地论文库的，请务必优先使用 search_pdf_database。如果该工具返回了图片，请务必仔细观察图片内容，结合文字给出极其专业的解答。\n"
            "2. 如果问题超出了本地库的范围，或者用户要求查询外部博客，请使用 tavily_search_results_json。\n"
            "请务必先思考用哪个工具最合适！如果都没查到，请诚实地说不知道。"
        )
    )

    return agent

my_agent = load_agent()


# ================= 界面与对话管理 =================

if "messages" not in st.session_state:
    st.session_state.messages =[]

# 渲染历史对话记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("Feel free to ask question about the paper"):
    
    # 记录并显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # AI 思考与回答
    with st.chat_message("assistant"):
        with st.spinner("等我睡五分钟"):
            
            # 把包含全部历史的 session_state 交给 Agent
            result = my_agent.invoke({"messages": st.session_state.messages})
            
            raw_content = result["messages"][-1].content
            
            # 1. 应对多模态列表返回格式
            if isinstance(raw_content, list):
                clean_answer = "\n".join([item.get("text", "") for item in raw_content if isinstance(item, dict) and "text" in item])
            else:
                clean_answer = str(raw_content)
                
            # 2. 终极清洗大法：修复被强制转义的换行符
            clean_answer = clean_answer.replace("\\n", "\n")
            
            # 3. 暴力砍掉 Google 泄漏的底层签名元数据 (Extras/Signature)
            if "', 'extras': {" in clean_answer:
                clean_answer = clean_answer.split("', 'extras': {")[0]
                # 顺手清理由于元组转字符串可能残余的头部符号
                if clean_answer.startswith("('"):
                    clean_answer = clean_answer[2:]
            
            answer = clean_answer
            
            # --------------------------------------------------------
            
            # 显示答案
            st.markdown(answer)
            
    # 记录 AI 的回答
    st.session_state.messages.append({"role": "assistant", "content": answer})

    
            
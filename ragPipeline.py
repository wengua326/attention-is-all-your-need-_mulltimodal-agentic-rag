__import__('pysqlite3')  # 1. 强行把我们刚才下载的最新版引擎加载进来
import sys               # 2. 召唤 Python 的底层模块管家
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3') # 3. 狸猫换太子！


from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnableLambda , RunnablePassthrough
from langchain_core.messages import SystemMessage , HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from base64 import b64decode
import embedding as emb
import pickle #准备解冻

import streamlit as st #webdevelopment


from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults


st.set_page_config(page_title = 'I love Kerian', layout= 'wide')
st.title('Agentic Multimodal Rag System for Attention is all your need')

@st.cache_resource #只运行一次
def load_agent() :
    def parse_docs(docs):
        b64 = []
        text = []
        for doc in docs :
            try :
                unwrapped_doc = pickle.loads(doc) #把byte 换回去
                text.append(unwrapped_doc)
            except Exception :
                unwrapped_image = doc.decode('utf-8')
                b64.append(unwrapped_image)
        return {'image' : b64 , 'texts' : text}

    @tool
    def search_pdf_database(query: str) -> list:
        """
        当你需要回答关于《Attention Is All You Need》论文、Transformer架构时，必须调用此工具去检索本地数据库。
        """
        docs = emb.retriever.invoke(query) 
        parsed_data = parse_docs(docs)
        
        # 准备一个空的“积木盒”
        content_blocks = []
        
        # 拼装文字和表格
        text_list = []
        if len(parsed_data['texts']) > 0:
            for element in parsed_data['texts']:
                element_type = str(type(element))
                if 'Table' in element_type or 'table' in element_type:
                    # 表格加上外框提示
                    text_list.append(f"\n--- [表格数据开始] ---\n{element.text}\n--- [表格数据结束] ---\n")
                else:
                    text_list.append(element.text)
                    
        context_text = '\n\n'.join(text_list)
        if not context_text:
            context_text = "没有找到相关的文字资料。"
            
        # 把文字作为第一块积木放进盒子里
        content_blocks.append({
            "type": "text", 
            "text": f"检索到的文字与表格信息如下：\n{context_text}"
        })

        # 拼装图片积木（让模型真正“看”到图片）
        if len(parsed_data['image']) > 0:
            for b64_img in parsed_data['image']:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                })
                
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
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

    # --- 6. 召唤终极 Agent ---
    agent = create_agent(
        model=llm, 
        tools=tools, 
        system_prompt=(
            "你是一个极其聪明、支持视觉处理的学术与网络助手。你现在有两个工具：\n"
            "1. 如果问题是关于你本地论文库的，请使用 search_pdf_database。如果该工具返回了图片，请务必仔细观察图片内容，结合文字给出极其专业的解答。\n"
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

    
            

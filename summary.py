from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chunking as ck

#prompt
prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additionnal comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}
"""

#summaryChain
text_prompt = ChatPromptTemplate.from_template(prompt_text)           #把prompt丢给ai
model = ChatGroq(temperature=0.5,model='llama-3.1-8b-instant')   #连接大模型
text_chain =text_prompt | model | StrOutputParser ()             #做一个pipeline

#summaryText
text_summaries = text_chain.batch(ck.texts,{'max_concurrency':1}) #.batch用来批量处理，maxconcurrency用来多并发处理

#summaryTable
table_html = [table.metadata.text_as_html for table in ck.tables]     #把tables 里面的html代码拿出来放进list
table_summaries = text_chain.batch(table_html,{'max_concurrency':1})


#summaryImage
from langchain_google_genai import ChatGoogleGenerativeAI
image_base64 = [im.metadata.image_base64 for im in ck.images] #把之前的image里面的base64码拿出来

prompt_template = """
Describe the image in detail. For context, 
the image is part of a research paper explaining the transformers 
architecture. Be specific about graphs, such as bar plots.
"""

# 因为有图片所以需要用list of message , from_template 只能处理文字
messages = [
    (
        "user",
        [
            {"type": "text", "text": prompt_template},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image_base64}"},
            },
        ],
    )
]

image_prompt = ChatPromptTemplate.from_messages(messages)
image_chain = image_prompt | ChatGoogleGenerativeAI (model='gemini-3.1-flash-lite-preview') | StrOutputParser()

#image_summaries = image_chain.batch(image_base64) 不能用这个不然会中limiterror
image_summaries = []
import time
for element in image_base64:
    try :
        res = image_chain.invoke({"image_base64": element})
        image_summaries.append(res)
        time.sleep(4)
        print('ok')
    except Exception as e :
       time.sleep(10)
       print(e)

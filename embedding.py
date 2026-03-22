# 1. Chroma 向量数据库
from langchain_community.vectorstores import Chroma
# 2. 内存存储库
from langchain_classic.storage import LocalFileStore
# 4. OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# 5. 多向量超级管理员
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

vectorstore = Chroma (collection_name='multi_modal_rag',
                      embedding_function=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview"),
                      persist_directory='./chroma_db') #把summary 存在local file
                      
store = LocalFileStore('./docstore') #存文件在local file
id_key = 'doc_id' #用来连接summary 和文件

retriever = MultiVectorRetriever (
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key, 
)


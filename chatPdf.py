from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Loader
loader = PyPDFLoader("unsu.pdf")
pages = loader.load_and_split()

# Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)

texts = text_splitter.split_documents(pages)

# Embeddings
embeddings_model = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings_model)

# Qusetion
question = "pdf 자료를 읽고, 김첨지의 직업을 알려줘"
llm = ChatOpenAI(streaming=True, temperature=0,callbacks=[StreamingStdOutCallbackHandler()])
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)

docs = retriever_from_llm.get_relevant_documents(query=question)

qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
result = qa_chain({"query": question})

print(result)
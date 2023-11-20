from dotenv import load_dotenv
load_dotenv()
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

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
question = "아내가 먹고 싶어하는 음식은 무엇이야?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)

docs = retriever_from_llm.get_relevant_documents(query=question)
print(len(docs))
print(docs)
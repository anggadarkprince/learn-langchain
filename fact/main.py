from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
#emb = embeddings.embed_query("hi there")
#print(emb)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200, # take at most 200 character to nearest separator, if 1 chunk is not enough, get next chunk to meet at least met the chunk_size
    chunk_overlap=100 # overlap chunk before (backward) 100 character
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter)

db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

results = db.similarity_search(
    "What is an interesting fact about the Englist language",
    k=2 # get 2 most relevant (similar)
)
for result in results:
    print("\n")
    print(result.page_content)
import os
import openai
import logging
import sys
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response.notebook_utils import display_response
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass.getpass("OpenAI API Key:")
openai.api_key = os.environ["OPENAI_API_KEY"]

# ChromaDB setup 
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")

embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    dimensions=3072,
)

# load documents
documents = SimpleDirectoryReader("./documents").load_data()


# initialize LLM + node parser
llm = OpenAI(model="gpt-4")
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)


# initialize storage context 
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = chromadb.ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
  documents, storage_context=storage_context, embed_model=embed_model
)


# retireve the top 10 most similar nodes using embeddings
vector_retriever = index.as_retriever(similarity_top_k=10)


# retireve the top 10 most similar nodes using bm25
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes
index.as_retriever(similarity_top_k=5)

hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")

query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
    llm=llm,
)


# Users Query
response = query_engine.query(
    "What is the impact of climate change on the ocean?"
)

display_response(response)

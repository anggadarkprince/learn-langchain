from typing import Any, Dict, List
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma

    def get_relevant_documents(self, query: str):
        # calculate embeddings for the 'query' string
        emb = self.embeddings.embed_query(query)

        # take embeddings and feed them into that
        # max_marginal_relevance_search_by_vectory
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8 # tollerance similar document
        )
    
    async def aget_relevant_documents(self):
        return []
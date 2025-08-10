from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore, PineconeRerank
from langchain_core.documents import Document

from settings import OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME, logger, DEBUG_LOGGING

class VectorStore:
    def __init__(self):
        # Initialize embeddings + Pinecone vector store
        logger.info("Initializing VectorStore with OpenAI embeddings and Pinecone")
        logger.debug(f"Using index name: {INDEX_NAME}")

        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set or empty")
        if not PINECONE_API_KEY:
            logger.error("PINECONE_API_KEY is not set or empty")

        try:
            self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
            logger.debug("OpenAI embeddings initialized successfully")

            self.store = PineconeVectorStore(
                index_name=INDEX_NAME,
                embedding=self.embeddings,
                pinecone_api_key=PINECONE_API_KEY
            )
            logger.info("Pinecone vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def add_documents(self, docs: List[Document]):
        try:
            logger.info(f"Adding {len(docs)} documents to vector store")

            if DEBUG_LOGGING:
                for i, doc in enumerate(docs):
                    metadata = doc.metadata or {}
                    logger.debug(f"Document {i+1}/{len(docs)}:")
                    logger.debug(f"  Title: {metadata.get('title', 'No title')}")
                    logger.debug(f"  Video URL: {metadata.get('video_url', 'No URL')}")
                    logger.debug(f"  Start seconds: {metadata.get('start_seconds', 'N/A')}")
                    logger.debug(f"  Content length: {len(doc.page_content)} chars")
                    logger.debug(f"  Content preview: {doc.page_content[:100]}...")

            self.store.add_documents(docs)
            logger.info(f"Successfully added {len(docs)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def search(self, query: str) -> List[Tuple[Document, float]]:
        try:
            logger.info(f"Searching vector store for: '{query}' (k=20)")

            results = self.store.similarity_search_with_score(query, k=20)
            logger.info(f"Initial search returned {len(results)} results")

            if not results:
                return []

            # Extract just the documents for reranking
            documents = [doc for doc, _ in results]

            # Rerank the documents and always return only the top 1
            reranker = PineconeRerank(
                model="bge-reranker-v2-m3",
                top_n=1,
                return_documents=True,
                pinecone_api_key=PINECONE_API_KEY
            )
            reranked_docs = reranker.compress_documents(list(documents), query=query)

            logger.info(f"Reranking returned {len(reranked_docs)} results")

            # Convert reranked documents to the expected return format (Document, float)
            # Use the relevance_score from reranking as the score
            final_results = []
            for doc in reranked_docs:
                relevance_score = doc.metadata.get('relevance_score', 0.0)
                final_results.append((doc, relevance_score))

            if DEBUG_LOGGING and final_results:
                for i, (doc, score) in enumerate(final_results):
                    metadata = doc.metadata or {}
                    logger.debug(f"Result {i+1}/{len(final_results)}:")
                    logger.debug(f"  Relevance Score: {score}")
                    logger.debug(f"  Title: {metadata.get('title', 'No title')}")
                    logger.debug(f"  Video URL: {metadata.get('video_url', 'No URL')}")
                    logger.debug(f"  Start seconds: {metadata.get('start_seconds', 'N/A')}")
                    logger.debug(f"  Content preview: {doc.page_content[:100]}...")

            return final_results
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise


vs = VectorStore()

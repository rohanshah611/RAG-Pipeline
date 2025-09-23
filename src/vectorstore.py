from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as LC_Pinecone
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    def __init__(self, pc_api_key:str, openai_api_key:str, index_name:str, embeddings, dimension:int, model:str, metric:int, region="us-east-1"):
        self.pc = Pinecone(api_key=self.pc_api_key)
        self.index_name = index_name
        self.embeddings = OpenAIEmbeddings(model=self.model, api_key=self.openai_api_key)
        self.dimension = dimension 
        self.metric = metric
        self.region = region

    def create_vector_db(self):
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud="aws", region=self.region)
            )
            print(f"Index '{self.index_name}' created!")

        else:
            # Connect to existing index
            index = self.pc.Index(self.index_name)
            print(f"Index '{self.index_name}' already exists and is ready.")

            
    def upsert_chunk(self, chunk_id: str, content: str):
        """
        Embed the given chunk content and upsert into Pinecone.
        """
        embedded_chunk = self.embeddings.embed_documents([content])[0]

        self.index.upsert([{
            "id": chunk_id,
            "values": embedded_chunk,
            "metadata": {
                "chunk_id": chunk_id,
                "type": "composite",
                "content": content,
                "text": content
            }
        }])

    def query_pinecone(self, query_text, top_k:int):
        """ 
        Query Pinecone index with a text query and return results.

        Args:
        index: Pinecone index object
        embeddings: Embedding model object
        query_text (str): Input query text
        top_k (int): Number of top results to fetch

        Returns:
            list: List of result dictionaries with id, content, score, and metadata
        """
    # Step 1: Convert query text to vector
        query_vector = self.embeddings.embed_query(query_text)

        # Step 2: Query Pinecone index
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=True
        )

        # Step 3: Format results
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                "id": match['id'],
                "content": match['metadata'].get('content', ''),
                "score": match['score'],
                "metadata": match['metadata']
            })

        return formatted_results


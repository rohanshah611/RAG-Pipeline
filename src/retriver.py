from pinecone import Pinecone

class Retriever:
    def __init__(self, index_name, model, openai_api_key, search_type: str = "similarity", k: int = 5):
        self.index_name=index_name,
        self.model = model
        self.openai_api_key = openai_api_key
        self.search_type = search_type
        self.k = k

    def create_retriver():
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(self.index_name)
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

        retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

    def query(self, query_text: str):
        index = pc.Index(index_name)
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        return self.retriever.get_relevant_documents(query_text)
from langchain_openai import ChatOpenAI

class RAGPipeline:
    def __init__(self, retriever, model="gpt-4o-mini", temperature=0.3):
        self.retriever = retriever
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def run(self, query):
        docs = self.retriever.retrieve(query)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Answer based only on the following context:\n{context}\nQuestion: {query}"
        return self.llm.predict(prompt)
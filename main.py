import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI,ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
load_dotenv()
s.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("hi")
    pdf_path = "download-3.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)
    llm = ChatOpenAI()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(), retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(), combine_docs_chain
    )
    query=("1.Give me the opening balance ?"
           "2.what was the start and end date? "
           "3. what is the person name?"
           "4. what is the closing balance?")
    #res = retrieval_chain.invoke({"input": "Give me the opening balance ? "})
    template = """ use the following pieces of context to answer the question at the end. if you dont know the answer, just say that you dont know, dont try to make up an answer.  
           {context}
           Question : {question}
           Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain = ({"context": vectorstore.as_retriever() | format_docs,
                  "question": RunnablePassthrough()} | custom_rag_prompt | llm)
    res = rag_chain.invoke(query)
    print(res)

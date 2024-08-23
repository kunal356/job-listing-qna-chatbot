import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import StreamlitCallbackHandler

load_dotenv()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text=text)
    return chunks


def create_vector_store(text_chunks, vector_store):
    vector_store.clear()
    vector_store.add_texts(texts=text_chunks)
    return vector_store


def main():

    model = ChatGroq(model='Llama3-8b-8192')
    prompt_template = """
    I have provided you various job alerts.
    Answer the questions based on the provided jobs only.
    Please provide the most accurate response based on the question.

    Context:\n {context}?\n
    Question: {question}
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=['context', 'question'])
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004")
    vector_store = AstraDBVectorStore(
        embedding=embeddings,
        namespace=os.environ['ASTRA_DB_NAMESPACE'],
        collection_name="test",
        token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
        api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
    )

    st.set_page_config(
        page_title="Job Listings Q&A Using Llama3", page_icon='🦜')

    st.title("🦜 Job Listings Q&A Using Llama3")
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process", accept_multiple_files=True)
        if st.button("Submit and Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs=pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                create_vector_store(text_chunks=text_chunks,
                                    vector_store=vector_store)
                st.success("Done")
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello!! I am your helpful assistant. How can I help you today??"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(
        placeholder="Enter your query about the job postings here.")

    if user_query:
        docs = vector_store.similarity_search(user_query, k=25)
        chain = load_qa_chain(model, chain_type='stuff',
                              verbose=True, prompt=prompt)
        st.session_state.messages.append(
            {"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            response = chain.invoke({"input_documents": docs, "question": user_query},
                                    return_only_outputs=True, callbacks=[streamlit_callback])
            st.session_state.messages.append(
                {"role": "assistant", "content": response['output_text']})
            st.write(response['output_text'])


if __name__ == '__main__':
    main()

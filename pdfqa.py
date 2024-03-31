from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def extract_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)


def create_knowledge_base(chunks):
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)


def get_user_question():
    return st.text_input("Ask a question about your PDF:")


def get_response(knowledge_base, user_question):
    docs = knowledge_base.similarity_search(user_question)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
    return response


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("PDF Content Query Interface")

    pdf = st.file_uploader("Upload PDF", type="pdf")
    if pdf is not None:
        text = extract_text_from_pdf(pdf)
        if text:  # check if text is not empty
            chunks = split_text_into_chunks(text)
            knowledge_base = create_knowledge_base(chunks)

            user_question = get_user_question()
            if user_question:
                response = get_response(knowledge_base, user_question)
                st.write(response)


if __name__ == '__main__':
    main()
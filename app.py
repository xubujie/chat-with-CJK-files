import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import time
from dotenv import load_dotenv
from config import CHROMA_SETTINGS, PERSIST_DIRECTORY

load_dotenv()

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0], history)
    history[-1][1] = ""
    
    for character in response:     
        history[-1][1] += character
        time.sleep(0.05)
        yield history

def infer(question, history):
    
    res = []
    for human, ai in history[:-1]:
        pair = (human, ai)
        res.append(pair)
    query = question
    result = qa({"query": query})
    return result["result"] + "\n\n Soruce:" + result["source_documents"][0].metadata["source"]

if __name__ == "__main__":
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    prompt_template = """使用以下内容来回答问题. 如果你不知道，请回答不知道，不要编撰答案.

    {context}

    问题: {question}
    答案:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    global qa
    qa = RetrievalQA.from_chain_type(
        ChatOpenAI(model="gpt-3.5-turbo", temperature=0), 
        chain_type="stuff", retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True)
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Markdown("## Chat with CJK PDF")        
            chatbot = gr.Chatbot([], elem_id="chatbot").style(height=400)
            question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
            submit_btn = gr.Button("Send Message")
        question.submit(add_text, [chatbot, question], [chatbot, question]).then(
            bot, chatbot, chatbot
        )
        submit_btn.click(add_text, [chatbot, question], [chatbot, question]).then(
            bot, chatbot, chatbot, queue=True)
    demo.queue(concurrency_count=4)
    demo.launch()


import streamlit as st
import pandas as pd
import uuid
import chromadb
from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Streamlit UI setup
st.title("Job Application Automation Tool")
st.write("Provide a job URL and generate a personalized email based on your portfolio.")

# Initialize Ollama LLM
@st.cache_resource
def load_model():
    return Ollama(model="llama2")

llm = load_model()

# Function definitions
def get_job_description(job_url):
    loader = WebBaseLoader(job_url)
    page_data = loader.load()[0].page_content
    return page_data

def create_chromadb_client(vectorstore_path):
    return chromadb.PersistentClient(vectorstore_path)

def create_portfolio_collection(client, portfolio_csv_path):
    collection = client.get_or_create_collection(name="portfolio")
    if collection.count() == 0:
        df = pd.read_csv(portfolio_csv_path)
        for _, row in df.iterrows():
            collection.add(
                documents=[row["Techstack"]],
                metadatas=[{"links": row["Links"]}],
                ids=[str(uuid.uuid4())]
            )
    return collection

def query_portfolio_collection(collection, skills):
    results = collection.query(query_texts=[skills], n_results=2)
    return [item.get('links', '') for item in results.get('metadatas', [[]])[0]]

def get_email_from_template(job_description, links, llm):
    prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}
        
        ### INSTRUCTION:
        You are sachin, a business development executive at xyz. xyz is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools. 
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
        process optimization, cost reduction, and heightened overall efficiency. 
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of xyz 
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase xyz's portfolio: {link_list}
        Remember you are sachin, BDE at xyz. 
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):
        
        """
    )
    chain_email = prompt_email | llm | StrOutputParser()
    try:
        return chain_email.invoke({"job_description": job_description, "link_list": links})
    except Exception as e:
        st.error(f"An error occurred while generating the email: {str(e)}")
        return "Unable to generate email due to an error. Please try again with a shorter job description or fewer links."

def add_new_entry_to_csv(portfolio_csv_path, techstack, link):
    new_data = {"Techstack": techstack, "Links": link}
    df = pd.read_csv(portfolio_csv_path)
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    df.to_csv(portfolio_csv_path, index=False)
    st.success(f"Added new entry: Techstack - {techstack}, Link - {link}")

# Set paths
vectorstore_path = 'vectorstore'
portfolio_csv_path = r'C:\Users\sachi\Downloads\my_portfolio.csv'

# Streamlit Inputs
st.sidebar.header("Inputs")
job_url = st.sidebar.text_input("Enter Job URL")

st.sidebar.subheader("Add to Portfolio CSV")
techstack_input = st.sidebar.text_input("Techstack (e.g., React, Node.js, MongoDB)")
link_input = st.sidebar.text_input("Portfolio Link (e.g., https://example.com/react-portfolio)")
add_entry = st.sidebar.button("Add Entry")

if add_entry and techstack_input and link_input:
    add_new_entry_to_csv(portfolio_csv_path, techstack_input, link_input)

if st.sidebar.button("Generate Email") and job_url:
    try:
        client = create_chromadb_client(vectorstore_path)
        collection = create_portfolio_collection(client, portfolio_csv_path)

        job_description = get_job_description(job_url)
        skills = job_description  # Simplified skill extraction

        links = query_portfolio_collection(collection, skills)

        email = get_email_from_template(job_description, links, llm)
        
        st.subheader("Generated Email")
        st.write(email)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
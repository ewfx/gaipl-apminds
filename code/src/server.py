import streamlit as st
import os
import PyPDF2
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.tokenizer._tokenizer.parallelism = False
#openai.api_key = os.getenv('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')
##client = OpenAI(api_key="k-proj-rFr5lQbIobjLj0dZERUuTSYGni_sS_pS6RMYAbYZzrWXuUUTc5helz2YRinbhUlR4SBC4pvRnMT3BlbkFJ_DSVf3O15RYQgAH1k8g3veep0bFDy5OoS8v0OwM4821TwVFLu6krUlsNr89zrqxLpvHCFi_KIA ")
# FAISS index for vector search
index = faiss.IndexFlatL2(384)  # 384 is the embedding size for MiniLM
doc_texts = []  # Store original document texts for reference
is_indexed = False
def process_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            pdf = PyPDF2.PdfReader(file, strict=False)
            text = "".join([page.extract_text() or "" for page in pdf.pages])
            return text
    except PyPDF2.errors.PdfReadError as e:
        st.error(f"PDF Error in {file_path}: {e}")
        return ""
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        return ""
# def process_pdf(file_path):
#     """Extract text from a PDF file."""
#     try:
#         with open(file_path, 'rb') as file:
#             pdf = PyPDF2.PdfReader(file)
#             text = "".join([page.extract_text() or "" for page in pdf.pages])
#             return text
#     except Exception as e:
#         st.error(f"Error processing {file_path}: {e}")
#         return ""

def index_documents(directory):
    """Index all PDFs in a directory."""
    global index, doc_texts, is_indexed
    if is_indexed:
        return
    
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            text = process_pdf(file_path)
            if text:
                embedding = model.encode(text, show_progress_bar=False)
                index.add(np.array([embedding]))
                doc_texts.append(text)
    
    is_indexed = True

# def search_documents(query):
#     """Search documents based on query."""
#     if not is_indexed or not doc_texts:
#         return "No documents indexed yet. Please upload PDFs."
    
#     query_embedding = model.encode(query, show_progress_bar=False)
#     D, I = index.search(np.array([query_embedding]), k=1)  # Top 1 result
    
#     if I[0][0] == -1:
#         return "No relevant document found."
    
#     return doc_texts[I[0][0]][:500]  # Return truncated text
def interpret_with_llm(query, context):
    """Use LLM to provide an intelligent interpretation of the search results."""
    try:
        response = client.chat.completions.create(
                 model="gpt-3.5-turbo",
                 messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise and informative interpretations of document search results."},
                {"role": "user", "content": f"Query: {query}\n\nContext: {context}\n\nProvide a clear and concise interpretation of the most relevant information from the context that answers the query."}
            ]
            )
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant that provides concise and informative interpretations of document search results."},
        #         {"role": "user", "content": f"Query: {query}\n\nContext: {context}\n\nProvide a clear and concise interpretation of the most relevant information from the context that answers the query."}
        #     ],
        #     max_tokens=300
        # )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"LLM Interpretation Error: {e}")
        return "Unable to generate interpretation."

def search_documents(query):
    """Search documents based on query."""
    if not is_indexed or not doc_texts:
        return "No documents indexed yet. Please upload PDFs."
    
    query_embedding = model.encode(query, show_progress_bar=False)
    D, I = index.search(np.array([query_embedding]), k=1)  # Top 1 result
    
    if I[0][0] == -1:
        return "No relevant document found."
    
    # Get the most relevant document
    most_relevant_doc = doc_texts[I[0][0]]
    
    # Use LLM to interpret the results
    interpretation = interpret_with_llm(query, most_relevant_doc[:1000])
    
    return interpretation

# Streamlit UI
st.title("Enterprise Document Query System")

uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    os.makedirs("pdfs", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join("pdfs", file.name), "wb") as f:
            f.write(file.read())
    st.success("Files uploaded successfully! Reindexing...")
    index_documents("pdfs")

query = st.text_input("Enter your query:")
if st.button("Search"):
    if query:
        response = search_documents(query)
        st.write("### Response:")
        st.write(response)
    else:
        st.warning("Please enter a query.")

import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # Fixed typo here
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def rank_resumes(job_description, resumes):
    documents = [job_description]  # Fixed variable name
    documents.extend(resumes)  # Changed to extend for clarity
    vectorizer = TfidfVectorizer()  # Fixed typo here
    vectors = vectorizer.fit_transform(documents)
    vectors_array = vectors.toarray()
    
    job_description_vector = vectors_array[0]
    resume_vectors = vectors_array[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

st.title("AI Resume Screening & Candidate Ranking System")  # Fixed typo

st.header("Job Description")
job_description = st.text_area("Enter the job description")

st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)  # Fixed missing value

if uploaded_files and job_description:  # Fixed variable name
    st.header("Ranking Resumes")
    
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)
    
    scores = rank_resumes(job_description, resumes)
    
    results = pd.DataFrame({
        "Resume": [file.name for file in uploaded_files], 
        "Score": scores  # Fixed variable name
    })
    
    results = results.sort_values(by="Score", ascending=False)  # Fixed variable name
    
    st.write(results)

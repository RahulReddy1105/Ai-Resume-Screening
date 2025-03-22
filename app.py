import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Function to apply background
def set_background():
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://source.unsplash.com/1600x900/?technology,abstract");
        background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n"
    return text if text else "(No extractable text)"

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Set Background
set_background()

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")
st.markdown("""<h4 style='color: white;'>Find the best candidates for your job opening effortlessly.</h4>""", unsafe_allow_html=True)

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    resumes = []
    extracted_texts = {}
    
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)
        extracted_texts[file.name] = text
    
    # Rank resumes
    scores = rank_resumes(job_description, resumes)
    
    # Create results dataframe
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    # Display results
    st.write(results.style.format({"Score": "{:.2f}"}))
    
    # Score threshold filter
    threshold = st.slider("Filter candidates by minimum similarity score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    filtered_results = results[results["Score"] >= threshold]
    
    if filtered_results.empty:
        st.warning("No resumes meet the selected threshold. Try lowering the filter value.")
    else:
        st.subheader("Filtered Candidates")
        st.write(filtered_results.style.format({"Score": "{:.2f}"}))
    
        # Download results as CSV
        csv_buffer = io.StringIO()
        filtered_results.to_csv(csv_buffer, index=False)
        st.download_button("Download Results as CSV", csv_buffer.getvalue(), "resume_rankings.csv", "text/csv")
    
    # Show extracted text for each resume
    with st.expander("Show Extracted Resume Texts"):
        for name, text in extracted_texts.items():
            st.subheader(f"Resume: {name}")
            st.text_area("Extracted Text", text, height=150)
    
    # Download extracted text data
    extracted_text_df = pd.DataFrame(list(extracted_texts.items()), columns=["Resume", "Extracted Text"])
    extracted_text_buffer = io.StringIO()
    extracted_text_df.to_csv(extracted_text_buffer, index=False)
    st.download_button("Download Extracted Texts", extracted_text_buffer.getvalue(), "extracted_texts.csv", "text/csv")

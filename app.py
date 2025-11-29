import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from io import StringIO
import json

load_dotenv()

# Configure Streamlit
st.set_page_config(
    page_title="AI Resume Screening Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize models
@st.cache_resource
def load_models():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return llm, embedder

llm, embedder = load_models()

# Utility functions
def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_candidate_info(resume_text, llm):
    """Extract candidate info using LLM"""
    prompt = PromptTemplate(
        input_variables=["resume"],
        template="""Extract the following information from the resume in JSON format:
{{
    "name": "candidate name",
    "email": "email if present",
    "phone": "phone if present",
    "experience_years": "total years of experience",
    "key_skills": ["skill1", "skill2", ...],
    "education": "degree and institution"
}}

Resume:
{resume}

Return only valid JSON."""
    )
    
    try:
        result = llm.invoke(prompt.format(resume=resume_text[:2000]))
        # Parse JSON from response
        json_match = re.search(r'\{[\s\S]*\}', result.content)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    return {
        "name": "Unknown",
        "email": "N/A",
        "phone": "N/A",
        "experience_years": "N/A",
        "key_skills": [],
        "education": "N/A"
    }

def extract_job_requirements(jd_text, llm):
    """Extract job requirements from JD"""
    prompt = PromptTemplate(
        input_variables=["jd"],
        template="""Extract job requirements from this job description in JSON format:
{{
    "position": "job title",
    "required_skills": ["skill1", "skill2", ...],
    "preferred_skills": ["skill1", "skill2", ...],
    "experience_required": "years of experience",
    "education_required": "education level"
}}

Job Description:
{jd}

Return only valid JSON."""
    )
    
    try:
        result = llm.invoke(prompt.format(jd=jd_text[:2000]))
        json_match = re.search(r'\{[\s\S]*\}', result.content)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    return {
        "position": "Unknown",
        "required_skills": [],
        "preferred_skills": [],
        "experience_required": "N/A",
        "education_required": "N/A"
    }

def calculate_match_score(resume_text, jd_text, candidate_info, job_req):
    """Calculate match score using embeddings and keyword matching"""
    # Embedding similarity
    resume_embedding = embedder.encode(resume_text[:1000], convert_to_tensor=True)
    jd_embedding = embedder.encode(jd_text[:1000], convert_to_tensor=True)
    embedding_score = float(cosine_similarity([resume_embedding], [jd_embedding])[0][0]) * 100
    
    # Keyword matching
    candidate_skills = set([s.lower() for s in candidate_info.get("key_skills", [])])
    required_skills = set([s.lower() for s in job_req.get("required_skills", [])])
    preferred_skills = set([s.lower() for s in job_req.get("preferred_skills", [])])
    
    if required_skills:
        required_match = len(candidate_skills & required_skills) / len(required_skills) * 100
    else:
        required_match = 0
    
    if preferred_skills:
        preferred_match = len(candidate_skills & preferred_skills) / len(preferred_skills) * 50
    else:
        preferred_match = 0
    
    # Weighted score
    final_score = (embedding_score * 0.4) + (required_match * 0.5) + (preferred_match * 0.1)
    return min(final_score, 100)

def generate_match_analysis(resume_text, jd_text, candidate_info, job_req, match_score, llm):
    """Generate detailed match analysis"""
    prompt = PromptTemplate(
        input_variables=["resume", "jd", "match_score"],
        template="""Based on the resume and job description provided, generate a brief hiring recommendation:

Resume Summary:
{resume}

Job Description:
{jd}

Match Score: {match_score}%

Provide:
1. Top matching qualifications (2-3 bullet points)
2. Missing qualifications (2-3 bullet points)
3. One-line recommendation (Proceed/Review/Not Recommended)

Keep response concise and actionable."""
    )
    
    try:
        result = llm.invoke(prompt.format(
            resume=resume_text[:1500],
            jd=jd_text[:1500],
            match_score=f"{match_score:.1f}"
        ))
        return result.content
    except:
        return "Analysis generation failed."

# Streamlit UI
st.markdown('<div class="header">📄 AI Resume Screening Agent</div>', unsafe_allow_html=True)
st.markdown("*Powered by LangChain, Gemini AI & Vector Embeddings*")
st.divider()

# Sidebar
with st.sidebar:
    st.title("Configuration")
    st.info("📌 This agent screens resumes against job descriptions using AI embeddings and semantic matching.")
    api_key = st.text_input("Google API Key", type="password", help="Get from https://makersuite.google.com/app/apikey")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Job Description")
    jd_source = st.radio("Input method:", ["Paste Text", "Upload PDF"], horizontal=True)
    
    if jd_source == "Paste Text":
        jd_text = st.text_area("Enter Job Description", height=250, placeholder="Paste the job description here...")
    else:
        jd_file = st.file_uploader("Upload Job Description PDF", type=["pdf"], key="jd_pdf")
        jd_text = extract_text_from_pdf(jd_file) if jd_file else ""
        if jd_text:
            st.success(f"✓ Extracted {len(jd_text)} characters")

with col2:
    st.subheader("📄 Resumes")
    resume_files = st.file_uploader(
        "Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple resume files"
    )
    st.caption(f"📊 {len(resume_files)} file(s) selected" if resume_files else "No files selected")

st.divider()

# Process button
if st.button("🚀 Start Screening", type="primary", use_container_width=True):
    if not jd_text:
        st.error("❌ Please provide a job description")
    elif not resume_files:
        st.error("❌ Please upload at least one resume")
    elif not os.environ.get("GOOGLE_API_KEY"):
        st.error("❌ Please provide Google API Key in sidebar")
    else:
        st.success("✓ Processing started...")
        
        # Extract job requirements
        with st.spinner("📖 Analyzing job description..."):
            job_req = extract_job_requirements(jd_text, llm)
        
        # Process resumes
        results = []
        progress_bar = st.progress(0)
        
        for idx, resume_file in enumerate(resume_files):
            with st.spinner(f"📝 Processing {resume_file.name}..."):
                resume_text = extract_text_from_pdf(resume_file)
                candidate_info = extract_candidate_info(resume_text, llm)
                match_score = calculate_match_score(resume_text, jd_text, candidate_info, job_req)
                analysis = generate_match_analysis(resume_text, jd_text, candidate_info, job_req, match_score, llm)
                
                results.append({
                    "filename": resume_file.name,
                    "candidate_name": candidate_info.get("name", "Unknown"),
                    "email": candidate_info.get("email", "N/A"),
                    "experience": candidate_info.get("experience_years", "N/A"),
                    "match_score": round(match_score, 2),
                    "skills_match": len(set([s.lower() for s in candidate_info.get("key_skills", [])]) & set([s.lower() for s in job_req.get("required_skills", [])])),
                    "analysis": analysis
                })
            
            progress_bar.progress((idx + 1) / len(resume_files))
        
        st.success("✓ Screening complete!")
        
        # Display results
        st.subheader("📊 Screening Results")
        
        # Sort by match score
        results_sorted = sorted(results, key=lambda x: x["match_score"], reverse=True)
        
        # Results table
        results_df = pd.DataFrame([
            {
                "Rank": idx + 1,
                "Candidate": r["candidate_name"],
                "Email": r["email"],
                "Experience": r["experience"],
                "Match Score": f"{r['match_score']}%",
                "Skills Match": r["skills_match"]
            }
            for idx, r in enumerate(results_sorted)
        ])
        
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        # Detailed analysis
        st.subheader("🔍 Detailed Analysis")
        for idx, result in enumerate(results_sorted, 1):
            with st.expander(f"#{idx} {result['candidate_name']} - Score: {result['match_score']}%"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Email:** {result['email']}")
                    st.write(f"**Experience:** {result['experience']}")
                    st.write(f"**File:** {result['filename']}")
                with col2:
                    st.metric("Match Score", f"{result['match_score']}%")
                    st.metric("Required Skills Matched", result['skills_match'])
                
                st.markdown("**Recommendation:**")
                st.write(result['analysis'])
        
        # Export results
        st.subheader("📥 Export Results")
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results (CSV)",
            data=csv,
            file_name="resume_screening_results.csv",
            mime="text/csv"
        )
        
        # Detailed JSON export
        json_export = json.dumps(results_sorted, indent=2)
        st.download_button(
            label="Download Detailed Results (JSON)",
            data=json_export,
            file_name="resume_screening_detailed.json",
            mime="application/json"
        )

st.divider()
st.caption("🔐 Your data is processed locally and not stored. API calls are made to Google Gemini only.")

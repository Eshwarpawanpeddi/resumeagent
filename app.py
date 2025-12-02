import streamlit as st
import pdfplumber
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import json

# Load environment variables
load_dotenv()

# Configure Streamlit
st.set_page_config(
    page_title="ResumeAI - Intelligent Screening",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN UI STYLING ---
st.markdown("""
    <style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* General Settings */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #2c3e50;
    }
    
    /* App Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 2rem 0;
        animation: fadeIn 1.2s ease-in-out;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #2980b9, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #57606f;
        margin-bottom: 2rem;
    }
    
    /* Card Container Style */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 15px;
        border-bottom: 2px solid #3498db;
        display: inline-block;
        padding-bottom: 5px;
    }
    
    /* Custom Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        padding: 0.7rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e1e4e8;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #3498db;
    }
    
    /* DataFrame Styling */
    [data-testid="stDataFrame"] {
        background: white;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Animations */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITY FUNCTIONS (Optimized) ---

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file safely."""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF {pdf_file.name}: {e}")
        return ""

def extract_candidate_info(resume_text, llm):
    """Extract candidate info using LLM."""
    prompt = PromptTemplate(
        input_variables=["resume"],
        template="""Extract the following information from the resume in JSON format:
{{
    "name": "candidate name",
    "email": "email if present",
    "phone": "phone if present",
    "experience_years": "total years of experience (numeric or string)",
    "key_skills": ["skill1", "skill2", ...],
    "education": "degree and institution"
}}

Resume:
{resume}

Return ONLY valid JSON."""
    )
    
    try:
        result = llm.invoke(prompt.format(resume=resume_text[:4000]))
        content = result.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        return json.loads(content)
    except Exception as e:
        return {
            "name": "Unknown",
            "email": "N/A",
            "experience_years": "N/A",
            "key_skills": [],
            "education": "N/A"
        }

def extract_job_requirements(jd_text, llm):
    """Extract job requirements from JD."""
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

Return ONLY valid JSON."""
    )
    
    try:
        result = llm.invoke(prompt.format(jd=jd_text[:4000]))
        content = result.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        return json.loads(content)
    except:
        return {
            "position": "Unknown",
            "required_skills": [],
            "preferred_skills": [],
            "experience_required": "N/A"
        }

def calculate_match_score(resume_text, jd_text, candidate_info, job_req, embeddings_model):
    """Calculate match score using Cloud Embeddings and Keyword Matching."""
    try:
        resume_vec = embeddings_model.embed_query(resume_text[:2000])
        jd_vec = embeddings_model.embed_query(jd_text[:2000])
        embedding_score = float(cosine_similarity([resume_vec], [jd_vec])[0][0]) * 100
    except Exception as e:
        embedding_score = 0
    
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
    
    final_score = (embedding_score * 0.4) + (required_match * 0.5) + (preferred_match * 0.1)
    return min(final_score, 100)

def generate_match_analysis(resume_text, jd_text, candidate_info, job_req, match_score, llm):
    """Generate detailed match analysis."""
    prompt = PromptTemplate(
        input_variables=["resume", "jd", "match_score"],
        template="""Generate a brief hiring recommendation based on this data:

Resume Summary:
{resume}

Job Description:
{jd}

Calculated Match Score: {match_score}%

Output Format:
1. Top Qualifications (2-3 bullets)
2. Missing Critical Skills (2-3 bullets)
3. Verdict (One line: Proceed / Review / Reject)

Keep it professional and concise."""
    )
    
    try:
        result = llm.invoke(prompt.format(
            resume=resume_text[:1500],
            jd=jd_text[:1500],
            match_score=f"{match_score:.1f}"
        ))
        return result.content
    except:
        return "Analysis unavailable."

# --- MAIN APP LAYOUT ---

# Hero Section
st.markdown("""
<div class="hero-container">
    <h1 class="main-title">ResumeAI</h1>
    <p class="subtitle">Intelligent Candidate Screening powered by Gemini Pro & Vector Embeddings</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2814/2814666.png", width=80)
    st.title("Settings")
    
    default_key = os.getenv("GOOGLE_API_KEY", "")
    api_key = st.text_input("🔑 Google API Key", value=default_key, type="password", help="Get from makersuite.google.com")
    
    st.markdown("---")
    st.caption("🚀 **Performance Mode:** Cloud Embeddings Active")
    st.caption("🛡️ **Security:** Data processed in-memory")

# Main Content Area - Input Section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-header">1. Job Description</div>', unsafe_allow_html=True)
    jd_source = st.radio("Input Method:", ["Paste Text", "Upload PDF"], horizontal=True, label_visibility="collapsed")
    
    jd_text = ""
    if jd_source == "Paste Text":
        jd_text = st.text_area("Paste JD here...", height=200, placeholder="Paste the full job description here...")
    else:
        jd_file = st.file_uploader("Upload JD PDF", type=["pdf"], key="jd_pdf")
        if jd_file:
            jd_text = extract_text_from_pdf(jd_file)
            st.success(f"✨ Extracted {len(jd_text)} characters")

with col2:
    st.markdown('<div class="section-header">2. Candidate Resumes</div>', unsafe_allow_html=True)
    resume_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    
    if resume_files:
        st.success(f"📂 {len(resume_files)} resumes ready to process")
    else:
        st.info("Upload one or more PDF resumes to begin.")

st.markdown('</div>', unsafe_allow_html=True) # End glass-card

# Action Button
col_center = st.columns([1, 2, 1])
with col_center[1]:
    process_btn = st.button("🚀 Analyze Candidates")

# Processing Logic
if process_btn:
    if not api_key:
        st.error("⚠️ Please enter your Google API Key in the sidebar.")
    elif not jd_text:
        st.error("⚠️ Please provide a Job Description.")
    elif not resume_files:
        st.error("⚠️ Please upload at least one resume.")
    else:
        # Progress Container
        progress_text = "Initializing AI Agents..."
        my_bar = st.progress(0, text=progress_text)
        
        try:
            # Initialize Models
            llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.0)
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
            
            # Step 1: Analyze JD
            my_bar.progress(10, text="Analyzing Job Requirements...")
            job_req = extract_job_requirements(jd_text, llm)
            
            results = []
            
            # Step 2: Process Resumes
            for idx, resume_file in enumerate(resume_files):
                progress_percent = 10 + int((idx / len(resume_files)) * 80)
                my_bar.progress(progress_percent, text=f"Screening {resume_file.name}...")
                
                # Extract
                resume_text = extract_text_from_pdf(resume_file)
                if not resume_text.strip():
                    continue
                    
                candidate_info = extract_candidate_info(resume_text, llm)
                
                # Score
                match_score = calculate_match_score(resume_text, jd_text, candidate_info, job_req, embeddings_model)
                
                # Analyze
                analysis = generate_match_analysis(resume_text, jd_text, candidate_info, job_req, match_score, llm)
                
                results.append({
                    "Rank": 0,
                    "Candidate": candidate_info.get("name", "Unknown"),
                    "Match Score": match_score,
                    "Email": candidate_info.get("email", "N/A"),
                    "Experience": candidate_info.get("experience_years", "N/A"),
                    "Skills Match": len(set([s.lower() for s in candidate_info.get("key_skills", [])]) & set([s.lower() for s in job_req.get("required_skills", [])])),
                    "Analysis": analysis,
                    "Filename": resume_file.name
                })
            
            my_bar.progress(100, text="Finalizing Results...")
            
            # Step 3: Display Results
            if results:
                results.sort(key=lambda x: x["Match Score"], reverse=True)
                for i, r in enumerate(results): r["Rank"] = i + 1
                
                st.balloons()
                
                # Results Section
                st.markdown("""
                <div class="hero-container" style="padding: 1rem 0;">
                    <h2 class="main-title" style="font-size: 2rem;">Analysis Results</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Summary Metrics
                top_candidate = results[0]
                m1, m2, m3 = st.columns(3)
                m1.metric("🏆 Top Candidate", top_candidate["Candidate"], f"{top_candidate['Match Score']:.1f}% Match")
                m2.metric("👥 Candidates Processed", len(results))
                m3.metric("⚡ Avg Match Score", f"{sum(r['Match Score'] for r in results)/len(results):.1f}%")
                
                # Table View
                df = pd.DataFrame(results)
                display_df = df[["Rank", "Candidate", "Match Score", "Experience", "Skills Match"]].copy()
                display_df["Match Score"] = display_df["Match Score"].apply(lambda x: f"{x:.1f}%")
                
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed Analysis Cards
                st.subheader("📝 Deep Dive Analysis")
                for res in results:
                    score_color = "#2ecc71" if res['Match Score'] >= 70 else "#f1c40f" if res['Match Score'] >= 50 else "#e74c3c"
                    
                    with st.expander(f"#{res['Rank']} {res['Candidate']} - {res['Match Score']:.1f}% Match"):
                        st.markdown(f"""
                        <div style="padding: 10px; border-left: 5px solid {score_color}; background: #f8f9fa; border-radius: 5px;">
                            <h4 style="margin:0; color: {score_color};">{res['Match Score']:.1f}% Match Score</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        c1, c2 = st.columns([2, 1])
                        with c1:
                            st.markdown("### 🤖 AI Verdict")
                            st.info(res['Analysis'])
                            st.markdown(f"**📧 Email:** {res['Email']}")
                            st.markdown(f"**📄 File:** {res['Filename']}")
                        with c2:
                            st.metric("Skills Matched", res['Skills Match'])
                            st.metric("Experience", res['Experience'])
                
                # Export
                st.markdown("---")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Full Report (CSV)", csv, "screening_report.csv", "text/csv")
                
            else:
                st.warning("No valid resumes found.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Tip: Check your API Key and connection.")

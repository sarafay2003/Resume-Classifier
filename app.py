import os
import streamlit as st
import pickle
from PyPDF2 import PdfReader
import re
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename="/tmp/streamlit.log", filemode="a")
logger = logging.getLogger(__name__)

# Ensure .streamlit directory for session state
os.makedirs(".streamlit", exist_ok=True)
os.environ["STREAMLIT_INSTALLATION_ID_DIR"] = "./.streamlit"

# Load models
try:
    rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
    tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
    rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
    tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))
    logger.debug("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    st.error(f"Error loading models: {e}")
    st.stop()

def pdf_to_text(file):
    try:
        reader = PdfReader(file)
        text = ''
        for page in range(len(reader.pages)):
            page_text = reader.pages[page].extract_text()
            if page_text:
                text += page_text
        logger.debug(f"Extracted text (first 100 chars): {text[:100]}...")
        return text
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        st.error(f"Error processing PDF: {e}")
        return ""

def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    return cleanText

def predict_category(resume_text):
    try:
        resume_text = cleanResume(resume_text)
        resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
        predicted_category = rf_classifier_categorization.predict(resume_tfidf)[0]
        logger.debug(f"Predicted category: {predicted_category}")
        return predicted_category
    except Exception as e:
        logger.error(f"Error predicting category: {e}")
        return "Error"

def job_recommendation(resume_text):
    try:
        resume_text = cleanResume(resume_text)
        resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
        recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
        logger.debug(f"Recommended job: {recommended_job}")
        return recommended_job
    except Exception as e:
        logger.error(f"Error recommending job: {e}")
        return "Error"

def extract_contact_no(text):
    pattern = re.findall(r'(?:\+92|0)?\s*\d{2,4}[\s\-]?\d{2,4}[\s\-]?\d{2,5}', text)
    cleaned_numbers = [re.sub(r'\s+', '', number.strip()) for number in pattern]
    return cleaned_numbers if cleaned_numbers else []

def extract_email(text):
    pattern = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return pattern if pattern else []

def extract_skills(text, skills_list):
    skills = []
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)
    return skills

def extract_education(text, education_list):
    education = []
    for degree in education_list:
        pattern = r"\b{}\b".format(re.escape(degree))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            education.append(degree)
    return education

def extract_name(text):
    pattern = r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}\b"
    name_matches = re.findall(pattern, text)
    return name_matches[0] if name_matches else None

skills_list = [
    "Python", "Java", "JavaScript", "SQL", "HTML", "CSS", "React", "Angular", "Django", "Node.js",
    "C#", "Ruby", "PHP", "Swift", "Kotlin", "C++", "Ruby on Rails", "Unity", "Vue.js", "Flask", "TensorFlow",
    "PyTorch", "Machine Learning", "Deep Learning", "API Development", "Git", "Version Control", " feltMongoDB",
    "PostgreSQL", "MySQL", "Docker", "Kubernetes", "Cloud Computing", "AWS", "Azure", "Google Cloud", "TypeScript",
    "GraphQL", "Firebase", "Android", "iOS", "C", "MATLAB", "Selenium", "Jenkins", "Bash", "Linux", "MacOS",
    "Windows", "Microservices", "Scrum", "Agile", "DevOps", "R", "Scala", "Hadoop", "Big Data", "Data Analysis", "C#"
]

education_list = [
    "High School Diploma", "Bachelor's Degree", "Bachelor of Science", "Bachelor of Arts",
    "Bachelor of Engineering", "Bachelor of Technology", "Bachelor of Commerce",
    "Bachelor of Science in Software Engineering", "Bachelor of Science in Computer Science",
    "Bachelor of Science in Information Technology", "Bachelor of Science in Data Science",
    "Master's Degree", "Master of Science", "Master of Arts", "Master of Engineering",
    "Master of Technology", "Master of Business Administration", "Master of Science in Data Science",
    "Master of Technology in Artificial Intelligence", "Master of Technology in Computer Science", "PhD", "Doctorate",
    "Associate's Degree", "Certification", "Diploma", "PMP", "Six Sigma", "Certified Information Systems Security Professional",
    "CISSP", "Data Science Certification", "Machine Learning Certification", "AWS Certified Solutions Architect",
    "Certified Scrum Master", "Certified Public Accountant", "Chartered Accountant", "MBA", "BSc", "BA", "MSc", "MA",
    "MEng", "MTech", "BCom", "BEng", "B.Tech", "MCA", "LLB", "MD", "MPhil", "BDS", "BAMS", "BPT", "MPT"
]

st.title("Resume Categorization System")
st.markdown("""
* Resume Categorization  
* Resume Job Recommendation  
* Resume Parsing
""")

uploaded_file = st.file_uploader("Upload your resume (PDF or TXT)", type=['pdf', 'txt'], key="file_uploader")

if uploaded_file:
    logger.debug(f"File uploaded: {uploaded_file.name}, size: {uploaded_file.size} bytes")
    try:
        if uploaded_file.name.endswith('.pdf'):
            with open(f"/tmp/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getvalue())
            text = pdf_to_text(f"/tmp/{uploaded_file.name}")
        elif uploaded_file.name.endswith('.txt'):
            text = uploaded_file.getvalue().decode('utf-8')
        else:
            st.error("Invalid file format. Upload a PDF or TXT file.")
            logger.error("Invalid file format uploaded")
            st.stop()

        if not text:
            st.error("No text extracted from the file.")
            logger.error("No text extracted from the file")
            st.stop()

        if st.button("Submit"):
            predicted_category = predict_category(text)
            recommended_job = job_recommendation(text)
            phone = extract_contact_no(text)
            email = extract_email(text)
            extracted_skills = extract_skills(text, skills_list)
            extracted_education = extract_education(text, education_list)
            name = extract_name(text)

            st.header("Results")
            st.write(f"**Category:** {predicted_category}")
            st.write(f"**Recommended Job:** {recommended_job}")

            st.write("**Phone Number(s):**")
            if phone:
                for number in phone:
                    st.write(f"- {number}")
            else:
                st.write("No phone number found.")

            st.write("**Email:**")
            if email:
                for email_addr in email:
                    st.write(f"- {email_addr}")
            else:
                st.write("No email found.")

            st.write("**Skills:**")
            if extracted_skills:
                for skill in extracted_skills:
                    st.write(f"- {skill}")
            else:
                st.write("No skills found.")

            st.write("**Education:**")
            if extracted_education:
                for edu in extracted_education:
                    st.write(f"- {edu}")
            else:
                st.write("No education details found.")

            if name:
                st.write(f"**Name:** {name}")

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        st.error(f"Error processing upload: {e}")
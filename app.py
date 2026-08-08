import nltk
import streamlit as st
import re
import pickle
import PyPDF2

nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def CleanResume(txt):
    txt = re.sub(r"http[s]?://\S+", " ", txt)
    txt = re.sub(r"[\r\n]+", " ", txt)
    txt = re.sub(r"[^\x00-\x7F]+", " ", txt)
    txt = re.sub(r"[^a-zA-Z ]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def main():
    # Title and subtitle
    st.title("📄 Resume Screening App")
    st.markdown("Upload your resume and let the app predict your career category.")

    # Sidebar instructions
    st.sidebar.header("Instructions")
    st.sidebar.write(
        "1. Upload a `.txt` or `.pdf` resume.\n"
        "2. The app will clean and process the text.\n"
        "3. You'll see the predicted category below."
    )

    # File uploader
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'], key="resume_uploader")

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                resume_text = read_pdf(uploaded_file)
            else:
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        # Clean and transform
        cleaned_resume = CleanResume(resume_text)
        input_feature = tfidf.transform([cleaned_resume]).toarray()

        # Predict
        prediction_id = clf.predict(input_feature)[0]

        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Developer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and Fitness",
            19: "PWO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        # Styled output
        st.success(f"🎯 Predicted Category: **{category_name}**")

        # Optional: show cleaned text
        with st.expander("See cleaned resume text"):
            st.text(cleaned_resume)

if __name__ == "__main__":
    main()

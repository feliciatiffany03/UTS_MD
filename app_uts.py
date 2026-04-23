import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG PAGE ---
st.set_page_config(page_title="Career Predictor Pro", layout="wide", page_icon="🎓")

# --- LOAD MODEL ---
current_dir = os.path.dirname(__file__)
path_status = os.path.join(current_dir, 'artifacts', 'model_status.pkl')
path_gaji = os.path.join(current_dir, 'artifacts', 'model_gaji.pkl')

@st.cache_resource
def load_models():
    return joblib.load(path_status), joblib.load(path_gaji)

model_status, model_gaji = load_models()

# --- SIDEBAR (LO 3: UI/UX Design) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("User Profile")
    st.info("Input data akademik Anda di kolom utama untuk memprediksi peluang karir.")
    st.markdown("---")
    st.write("**Student ID:** #12345")
    st.write("**Major:** Computer Science")

# --- MAIN INTERFACE ---
st.title('🎓 Student Career Prediction System')
st.write("Aplikasi ini menggunakan Machine Learning untuk memprediksi status penempatan kerja.")

# --- FORM (LO 3: Intuitif UI) ---
with st.form("main_form"):
    tab1, tab2 = st.tabs(["📊 Academic & Skills", "📝 Personal & Others"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            cgpa = st.number_input("Current CGPA", 0.0, 10.0, 8.5, step=0.1)
            tech_score = st.slider("Technical Skill Score", 0, 100, 80)
            soft_score = st.slider("Soft Skill Score", 0, 100, 75)
        with col2:
            ssc = st.number_input("SSC (10th) Percentage", 0.0, 100.0, 80.0)
            hsc = st.number_input("HSC (12th) Percentage", 0.0, 100.0, 80.0)
            entrance = st.number_input("Entrance Exam Score", 0.0, 100.0, 75.0)

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            gender = st.selectbox("Gender", ["Male", "Female"])
            work_exp = st.number_input("Work Experience (Months)", 0, 100, 6)
            interns = st.number_input("Internship Count", 0, 5, 1)
        with col4:
            extra = st.radio("Extracurricular Activities", ["Yes", "No"])
            backlogs = st.number_input("Backlogs Count", 0, 10, 0)
            attendance = st.slider("Attendance Percentage", 0, 100, 95)

    submit = st.form_submit_button("Analyze My Career Path")

# --- LOGIC & VISUALIZATION (LO 1 & LO 2) ---
if submit:
    # Siapkan Data (Sesuaikan 16 kolom B.csv)
    data = {
        'student_id': 1, 'gender': gender, 'ssc_percentage': ssc, 'hsc_percentage': hsc,
        'degree_percentage': 75.0, 'cgpa': cgpa, 'entrance_exam_score': entrance,
        'technical_skill_score': tech_score, 'soft_skill_score': soft_score,
        'internship_count': interns, 'live_projects': 2, 'work_experience_months': work_exp,
        'certifications': 2, 'attendance_percentage': attendance, 'backlogs': backlogs,
        'extracurricular_activities': extra
    }
    df = pd.DataFrame([data])
    
    # Samakan urutan kolom dengan model
    if hasattr(model_status, 'feature_names_in_'):
        df = df[model_status.feature_names_in_]

    # Prediksi
    status = model_status.predict(df)[0]

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.subheader("Prediction Result")
        if status == 0:
            st.success("### PLACED 🎉")
            gaji = model_gaji.predict(df)[0]
            st.metric("Estimated Salary", f"{gaji:.2f} LPA")
        else:
            st.error("### NOT PLACED ❌")
            st.write("Coba tingkatkan skor teknis atau kehadiran Anda.")

    with res_col2:
        st.subheader("Skill Analysis")
        # Visualisasi Sederhana (LO 3: Data Visualization)
        chart_data = pd.DataFrame({
            'Category': ['CGPA', 'Technical', 'Soft Skill', 'Attendance'],
            'Score': [cgpa*10, tech_score, soft_score, attendance]
        })
        fig, ax = plt.subplots()
        sns.barplot(x='Category', y='Score', data=chart_data, ax=ax, palette='viridis')
        st.pyplot(fig)
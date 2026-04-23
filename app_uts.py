import streamlit as st
import joblib
import pandas as pd

model_status = joblib.load('model_status.pkl')
model_gaji = joblib.load('model_gaji.pkl')

def main():
    st.title('Student Career Prediction Dashboard')

    # Input dari user
    cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0)
    tech_score = st.number_input("Technical Skill Score", 0, 100, 75)
    # Ganti 'Experience Score' jadi 'work_experience_months' sesuai kolom B.csv
    work_exp = st.number_input("Work Experience (Months)", 0, 100, 6)
    
    # Tambahin dua input ini yang tadi kurang
    ssc = st.number_input("SSC Percentage", 0.0, 100.0, 80.0)
    hsc = st.number_input("HSC Percentage", 0.0, 100.0, 80.0)
    
    gender = st.radio("Gender", ["Male", "Female"])
    extra = st.radio("Extracurricular Activities", ["Yes", "No"])
    
    # Susun data sesuai urutan dan nama kolom saat training
    data = {
        'cgpa': cgpa, 
        'gender': gender,
        'technical_skill_score': tech_score, 
        'work_experience_months': work_exp, # Nama kolom harus pas
        'ssc_percentage': ssc,              # Kolom tambahan
        'hsc_percentage': hsc,              # Kolom tambahan
        'extracurricular_activities': extra
    }
    
    df = pd.DataFrame([data])
    
    if st.button("Make Prediction"):
        # Prediksi status dulu
        status = model_status.predict(df)[0]
        
        if status == 0:
            # Kalau Placed, baru prediksi gaji
            gaji = model_gaji.predict(df)[0]
            st.success(f"Result: PLACED 🎉")
            st.metric("Estimated Salary", f"{gaji:.2f} LPA")
        else:
            st.error("Result: NOT PLACED ❌")

if __name__ == "__main__":
    main()
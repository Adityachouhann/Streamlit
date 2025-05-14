import numpy as np
import pickle
import pandas as pd
import streamlit as st 

pickle_in = open(r"C:\Users\ajcho\OneDrive\Desktop\full stack data science and AI\BY SELF\ML\ML self\linear_salary_model.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome All"
    
def predict_note_authentication(YearsExperience):
   
    prediction=classifier.predict([[YearsExperience]])
    print(prediction)
    return prediction

def main():
    st.title("Salary_cheak")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Salary ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    YearsExperience = st.text_input("YearsExperience","")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(eval(YearsExperience))
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("This is about Salary")

if __name__=='__main__':
    main()
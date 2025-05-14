import numpy as np 
import pandas as pd
import pickle 
import streamlit as st

pickle_in=open(r"C:\Users\ajcho\OneDrive\Desktop\full stack data science and AI\BY SELF\ML\ML self\Linear_housing_model.pkl", "rb")
classifier=pickle.load(pickle_in)

def predict_housing(MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude)
    predict=classifier.predict([[MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude]])
    return prediction

def main():
    st.title('House Price Prediction')

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Salary ML App</h2>
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)

    MedInc=st.text_input('Enter the Medinc','')
    HouseAge=st.text_input('Enter the HouseAge','')
    AveRooms=st.text_input('Enter the AveRooms','')
    AveBedrms=st.text_input('Enter the AveBedrms','')
    Population=st.text_input('Enter the Population','')
    AveOccup=st.text_input('Enter the AveOccup','')
    Latitude=st.text_input('Enter the Latitude','')
    Longitude=st.text_input('Enter the Longitude','')

    result=''
    if st.button('Predict'):
        try:
            result=predict_housing(float(MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude))
            st.success(f'The predicted house price is ${result[0]:,.2f}')
        except ValueError:
            st.error('Please enter a valid number.')

        if st.button('About'):
            st.text("Let's Learn!")
            st.text('This app predicts house price.')

        if __name__=='__main__':
            main()

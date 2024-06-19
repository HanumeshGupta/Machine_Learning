import numpy as np
import pickle
import streamlit as st

# loading the train model 
loaded_model = pickle.load(open('D:/Visual Studio Code/ML/Youtube_ML_Code/Diabetes_Prediction/trained_model.sav','rb'))



#Creating the function 
def diabetes_prediction(input_data):
    #Changing this data into an array
    input_data = np.asarray(input_data)

    #Changing its shape
    input_data = input_data.reshape(1,-1)


    prediction = loaded_model.predict(input_data)

    if (prediction[0] == 0):
        return "Not Diabetic"
    else:
        return "Diabetic"
    
    
def main():
    
    # Giving the title 
    st.title('Diabetes Prediction Web App')
    
    #getting the input data from user 
    Pregnancies = st.text_input('Number of Pregnancies: ')
    Glucose = st.text_input('Glucose Level: ')
    BloodPressure = st.text_input('Blood Presure: ')
    SkinThickness= st.text_input('Skin Thickness: ')
    Insulin= st.text_input('Insuline Level: ')
    BMI	= st.text_input('Body Mass Index: ')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree: ')
    Age = st.text_input('Age: ')
    
    #Code in prediction
    diagnosis = ''
    
    
    #Creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    
    st.success(diagnosis)
    
    
    
    
    
if __name__ =='__main__':
    main()
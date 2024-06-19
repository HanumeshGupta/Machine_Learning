import pickle
import streamlit as st 
import numpy as np

# Load Model
load_model = pickle.load(open('D:/Visual Studio Code/ML/Youtube_ML_Code/Heart_Disease_Preiction/Heart_Prediction_model.sav', 'rb'))

# Function for prediction 
def heart_prediction(input_data):
    # Change the input data to a numpy array
    input_data_as_array = np.asarray(input_data, dtype=float)
    # Reshape the numpy array as the data model we trained
    data_reshape = input_data_as_array.reshape(1, -1)

    pred = load_model.predict(data_reshape)

    if pred == 1:
        return "Heart has some defect."
    else:
        return "Your heart is perfect :)"

def main():
    st.title("Heart Disease Prediction")
    
    age = st.text_input('Age: ')
    sex = st.text_input('Sex: ') 
    cp = st.text_input('chest pain type: ')
    trestbps = st.text_input('resting blood pressure: ')
    chol = st.text_input('serum cholestoral in mg/dl: ')
    fbs = st.text_input('fasting blood sugar: ')
    restecg = st.text_input('resting electrocardiographic results: ')
    thalach = st.text_input('maximum heart rate achieved: ')
    exang = st.text_input('exercise induced angina: ')
    oldpeak = st.text_input('oldpeak (ST depression induced by exercise relative to rest): ')
    slope = st.text_input('the slope of the peak exercise ST segment: ')
    ca = st.text_input('number of major vessels: ')
    thal = st.text_input('thal: ')
    
    result = ''
    
    # Creating a button for prediction
    if st.button('Heart Disease Result'):
        try:
            input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            # Convert input data to floats
            input_data = [float(i) for i in input_data]
            result = heart_prediction(input_data)
        except ValueError:
            result = "Please enter valid numeric values"
        
    st.success(result)

if __name__ == '__main__':
    main()

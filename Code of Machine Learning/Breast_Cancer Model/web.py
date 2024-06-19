import pickle
import numpy as np
import streamlit as st

# Load Model 
load_model = pickle.load(open('D:/Visual Studio Code/ML/Youtube_ML_Code/Breast_Cancer Model/Breast_Cancer.sav', 'rb'))

# Predicting Model
def predict_model(input_data):
    input_data = np.asarray(input_data, dtype=float)
    input_data = input_data.reshape(1, -1)
    pred = load_model.predict(input_data)
    if pred[0] == 0:
        return "Malignant"
    else:
        return 'Benign'

def main():
    st.title("Breast Cancer Detection")
    
    # Getting the input data
    radius_mean = st.text_input('radius_mean')
    texture_mean = st.text_input('texture_mean')
    perimeter_mean = st.text_input('perimeter_mean')
    area_mean = st.text_input('area_mean')
    smoothness_mean = st.text_input('smoothness_mean')
    compactness_mean = st.text_input('compactness_mean')
    concavity_mean = st.text_input('concavity_mean')
    concave_points_mean = st.text_input('concave_points_mean')
    symmetry_mean = st.text_input('symmetry_mean')
    fractal_dimension_mean = st.text_input('fractal_dimension_mean')
    radius_se = st.text_input('radius_se')
    texture_se = st.text_input('texture_se')
    perimeter_se = st.text_input('perimeter_se')
    area_se = st.text_input('area_se')
    smoothness_se = st.text_input('smoothness_se')
    compactness_se = st.text_input('compactness_se')
    concavity_se = st.text_input('concavity_se')
    concave_points_se = st.text_input('concave_points_se')
    symmetry_se = st.text_input('symmetry_se')
    fractal_dimension_se = st.text_input('fractal_dimension_se')
    radius_worst = st.text_input('radius_worst')
    texture_worst = st.text_input('texture_worst')
    perimeter_worst = st.text_input('perimeter_worst')
    area_worst = st.text_input('area_worst')
    smoothness_worst = st.text_input('smoothness_worst')
    compactness_worst = st.text_input('compactness_worst')
    concavity_worst = st.text_input('concavity_worst')
    concave_points_worst = st.text_input('concave_points_worst')
    symmetry_worst = st.text_input('symmetry_worst')
    fractal_dimension_worst = st.text_input('fractal_dimension_worst')
    
    # Code in prediction
    diagnosis = ''
    
    # Creating a button for prediction
    if st.button('Breast Cancer Test Result'):
        try:
            input_data = [
                radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
                fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
                smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
                symmetry_worst, fractal_dimension_worst
            ]
            # Convert input data to floats
            input_data = [float(i) for i in input_data]
            diagnosis = predict_model(input_data)
        except ValueError:
            diagnosis = "Please enter valid numeric values"
    
    st.success(diagnosis)

if __name__ == '__main__':
    main()

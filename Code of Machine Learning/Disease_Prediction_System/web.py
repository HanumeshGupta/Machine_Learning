import numpy as np
import pickle 
import streamlit as st
from streamlit_option_menu import option_menu



Diabetes = pickle.load(open('Diabetes_Prediction_model.sav','rb'))
Heart = pickle.load(open('D:/Visual Studio Code/ML/Youtube_ML_Code/Disease_Prediction_System/Heart_Prediction_model.sav','rb'))
parkinson = pickle.load(open('D:/Visual Studio Code/ML/Youtube_ML_Code/Disease_Prediction_System/Parkinson_Disease_Detection.sav','rb'))

# Sidebar for nevigation

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ["Diabetes Prediction","Heart Disease Prediction", "Parkinson's Disease Prediction"],
                          icons = ['activity','heart','person'],
                          default_index = 0 )




# Diabetes Dataset

if selected == 'Diabetes Prediction':
    
    
    # Creating the function 
    def diabetes_prediction(input_data):
        #Changing this data into an array
        input_data = np.asarray(input_data)

        #Changing its shape
        input_data = input_data.reshape(1,-1)


        prediction = Diabetes.predict(input_data)

        if (prediction[0] == 0):
            return "Not Diabetic"
        else:
            return "Diabetic"
        
        
        
    # Giving the title 
    st.title('Diabetes Prediction Web App')
        
    col1,col2,col3 = st.columns(3)
        
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies: ')
        
    with col2:
        Glucose = st.text_input('Glucose Level: ')
        
    with col3:
        BloodPressure = st.text_input('Blood Presure: ')
            
        
    with col1:
        SkinThickness= st.text_input('Skin Thickness: ')
        
    with col2:
        Insulin= st.text_input('Insuline Level: ')
        
    with col3:
        BMI	= st.text_input('Body Mass Index: ')
        
        
            
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree: ')
            
    with col2:
        Age = st.text_input('Age: ')
            
            
    diagnosis = ''
        
        
    #Creating a button for prediction
        
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
        
     
# #Heart Disease
if selected == 'Heart Disease Prediction':

    def heart_prediction(input_data):
        # Change the input data to a numpy array
        input_data_as_array = np.asarray(input_data, dtype=float)
        # Reshape the numpy array as the data model we trained
        data_reshape = input_data_as_array.reshape(1, -1)

        pred = Heart.predict(data_reshape)

        if pred == 1:
            return "Heart has some defect."
        else:
            return "Your heart is perfect :)"

    st.title("Heart Disease Prediction")
    
    col1,col2,col3 = st.columns(3)
    
    with col1:    
        age = st.text_input('Age: ')
    with col2:
        sex = st.text_input('Sex: ') 
    with col3:
        cp = st.text_input('chest pain type: ')
    
    with col1:
        trestbps = st.text_input('resting blood pressure: ')
    with col2:
        chol = st.text_input('serum cholestoral in mg/dl: ')
    with col3:
        fbs = st.text_input('fasting blood sugar: ')
    
    
    with col1:
        restecg = st.text_input('resting electrocardiographic results: ')
    with col2:
        thalach = st.text_input('maximum heart rate achieved: ')
    with col3:
        exang = st.text_input('exercise induced angina: ')
    
    
    with col1:
        oldpeak = st.text_input('oldpeak (ST depression induced by exercise relative to rest): ')
    with col2:
        slope = st.text_input('the slope of the peak exercise ST segment: ')
    with col3:
        ca = st.text_input('number of major vessels: ')
    
    
    with col1:
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
    
    
    
# # parkinson Disease
if selected == "Parkinson's Disease Prediction":

    def parkinson_disease(input_data):
        input_data = np.asarray(input_data)
        input_data = input_data.reshape(1,-1)

        # input_data = scaler.transform(input_data)
        pred = parkinson.predict(input_data)

        if pred[0] == 1:
            return 'The paisent has Parkinsons Disease'

        if pred[0] == 0:
            return 'The Paisent is healty'
        
        

    st.title("Parkinson's Disease Prediction")
    
    
    col1,col2,col3 = st.columns(3)
    
    
    with col1:
        MDVP_Fo = st.text_input('MDVP : Fo(Hz) :- ')
    with col2:
        MDVP_Fhi = st.text_input("MDVP : Fhi(Hz) :- ")
    with col3:
        MDVP_Flo = st.text_input("MDVP : Flo(Hz) :- ")
        
        
        
    with col1:
        MDVP_Jitter = st.text_input("MDVP : Jitter(%) :- ")
    with col2:
        MDVP_Jitter_Abs = st.text_input("MDVP : Jitter(Abs) :- ")     
    with col3:
        MDVP_RAP = st.text_input("MDVP : RAP :- ")
        
        
        
    with col1:
        MDVP_PPQ = st.text_input("MDVP : PPQ :- ")
    with col2:
        Jitter_DDP = st.text_input("Jitter : DDP :- ")    
    with col3:
        MDVP_Shimmer =st.text_input("MDVP : Shimmer :- ")
    
    
    with col1:
        MDVP_Shimmer_dB = st.text_input("MDVP : Shimmer(dB) :- ")   
    with col2:
        Shimmer_APQ3 =st.text_input("Shimmer : APQ3 :- ")
    with col3:
        Shimmer_APQ5 =st.text_input("Shimmer : APQ5 :- ")
    
    with col1:
        MDVP_APQ =st.text_input("MDVP : APQ :- ")
    with col2:
        Shimmer_DDA = st.text_input("Shimmer : DDA :- ")
    with col3:
        NHR = st.text_input("NHR:- ")
    
    
    with col1:
        HNR = st.text_input("HNR:- ")
    with col2:
        RPDE  =st.text_input("RPDE:- ")
    with col3:
        DFA  = st.text_input("DFA:- ")
    
    
    with col1:
        spread1 = st.text_input("spread1:- ")
    with col2:
        spread2 = st.text_input("spread2:- ")   
    with col3:
        D2 =st.text_input("D2:- ")
    
    
    with col1:
        PPE = st.text_input("PPE:- ")
        
        
    result =''
    
    # creating the button for prediction
    
    if st.button("Parkinson's Disease Result"):
        
        try:
            input_data = [MDVP_Fo,MDVP_Fhi,MDVP_Flo,MDVP_Jitter,MDVP_Jitter_Abs,MDVP_RAP,MDVP_PPQ,
                          Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,
                          MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]
            input_data = [float(i) for i in input_data]
            result = parkinson_disease(input_data)
        except:
            result = "Please enter valid numeric values"
            
    st.success(result)
    
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64



# Setting page congiuration
st.set_page_config(page_title = 'Cars_price_Prediction',layout='wide') 

# Load the encoder
with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# 1. Load the pre-trained model
with open('ext.pkl', 'rb') as file:
    ext = pickle.load(file)
 

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg("C:/Users/Kastromani/Pictures/car.png")

st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #9899AA;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color:black;'>CarsDekho - " \
"Cars Price Prediction</h1>", unsafe_allow_html=True)


df = pd.read_csv('cars_dekho.csv')


# Extract unique fuel-type options from the DataFrame
Fuel_Type= df["Fuel_Type"].unique().tolist()
selected_fuel = st.sidebar.selectbox("Select Fuel Type", Fuel_Type)


# Extract unique bodytype-type options from the DataFrame
Body_Type = df["Body_Type"].unique().tolist()
selected_body = st.sidebar.selectbox("Select Body Type", Body_Type)

# Extract unique Manufacturer options from the DataFrame
Manufacturer = df["Manufacturer"].unique().tolist()
selected_Manufacturer = st.sidebar.selectbox("Select Manufacturer", Manufacturer)

# Extract unique Model options from the DataFrame
Model  = df["Model"].unique().tolist()
selected_Model  = st.sidebar.selectbox("Select Model", Model )

# Extract unique Transmission_Type  options from the DataFrame
Transmission_Type  = df["Transmission_Type"].unique().tolist()
selected_Transmission_Type  = st.sidebar.selectbox("Select Transmission_Type ", Transmission_Type)

Seats = st.selectbox("Seats", options=[5, 7, 6, 4, 8, 9, 2, 10])

Car_Age = st.slider("Car_Age", 1, 40, 5)

Kms_Driven = st.slider("Kilometers_Driven", 10000, 500000, 10000)

Owner_No = st.selectbox("Owner_Number", options=[0,1,2,3,4,5])

Mileage = st.slider('Mileage(kmpl)',10.0, 50.0, 0.1)         

Max_Power = st.slider('Max_Power(bhp)',15.0, 510.0, 0.1)

Engine_Displacement= st.slider('Engine_Displacement(cc)',998, 5000, 1)

if st.button("Predict"):


        user_input = pd.DataFrame({
            'Fuel_Type': [selected_fuel],
            'Body_Type':[selected_body],
            'Kms_Driven':np.cbrt([Kms_Driven]),
            'Owner_No':[Owner_No],
            'Manufacturer':[selected_Manufacturer],
            'Model':[selected_Model],
            'Engine_Displacement(cc)':np.cbrt([Engine_Displacement]),
            'Transmission_Type':[selected_Transmission_Type],
            'Mileage(kmpl)':np.cbrt([Mileage]),
            'Max_Power(bhp)':np.cbrt([Max_Power]),
            'Seats':[Seats],
            'Car_Age':[Car_Age]
        })
        st.write(user_input)
        

    
        # Encode categorical features
        encoded_user_array = encoder.transform(user_input[['Fuel_Type','Body_Type','Manufacturer','Model','Transmission_Type']])
        encoded=encoded_user_array.toarray()

        # Convert to DataFrame
        encoded_user_df = pd.DataFrame(encoded, columns = encoder.get_feature_names_out())
        st.write(encoded_user_df)


        # Combine continuous variables with encoded categorical features
        final_user_df = pd.concat([user_input[['Kms_Driven','Owner_No','Engine_Displacement(cc)','Mileage(kmpl)','Max_Power(bhp)','Seats','Car_Age']], encoded_user_df], axis=1)
        final_user_df

        prediction = ext.predict(final_user_df)
        st.write(round(prediction[0],2))

   

    
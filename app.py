import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load pre-trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('Gradient_Boosting_Regressor.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

st.title('Laptop Price Prediction')

# Create input fields for each feature
company_options = ['Xiaomi', 'Asus', 'Huawei', 'Fujitsu', 'Lenovo', 'Samsung', 'Google', 'Toshiba', 'HP', 'Microsoft', 'Dell', 'Chuwi', 'LG', 'Vero', 'Apple', 'Mediacom', 'MSI', 'Acer', 'Razer', 'Other']
company = st.selectbox('Company', company_options)

os_options = ['Mac OS X', 'Windows 10 S', 'Android', 'macOS', 'Chrome OS', 'Windows 10', 'Windows 7', 'Linux', 'No OS']
os = st.selectbox('Operating System', os_options)

cpu_brand_options = ['Samsung', 'AMD', 'Intel', 'Other']
cpu_brand = st.selectbox('CPU Brand', cpu_brand_options)

gpu_brand_options = ['ARM', 'Intel', 'AMD', 'Nvidia', 'Other']
gpu_brand = st.selectbox('GPU Brand', gpu_brand_options)

type_name_options = ['2 in 1 Convertible', 'Netbook', 'Workstation', 'Ultrabook', 'Gaming', 'Notebook']
type_name = st.selectbox('Type', type_name_options)

pri_memory_type_options = ['Hybrid', 'Flash', 'HDD', 'SSD']
pri_memory_type = st.selectbox('Primary Memory Type', pri_memory_type_options)

sec_memory_type_options = ['SSD', 'Hybrid', 'HDD', 'None']
sec_memory_type = st.selectbox('Secondary Memory Type', sec_memory_type_options)

inches = st.number_input('Screen Size (inches)', min_value=10.0, max_value=20.0, value=15.6, step=0.1)
weight_kg = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=2.0, step=0.1)
pri_memory_amount_gb = st.number_input('Primary Memory Amount (GB)', min_value=8, max_value=2000, value=256, step=32)
sec_memory_amount_gb = st.number_input('Secondary Memory Amount (GB)', min_value=0, max_value=2000, value=0, step=128)
cpu_ghz = st.number_input('CPU GHz', min_value=0.9, max_value=5.0, value=2.5, step=0.1)
screen_height = st.number_input('Screen Height', min_value=600, max_value=2160, value=1080, step=1)
screen_width = st.number_input('Screen Width', min_value=800, max_value=3840, value=1920, step=1)
ram_gb = st.number_input('RAM (GB)', min_value=2, max_value=64, value=8, step=2)

# Create a dictionary to hold feature values
features = {f'Company_{comp}': 1 if company == comp else 0 for comp in company_options}
features.update({f'OpSys_{o}': 1 if os == o else 0 for o in os_options})
features.update({f'CpuBrand_{brand}': 1 if cpu_brand == brand else 0 for brand in cpu_brand_options})
features.update({f'GpuBrand_{brand}': 1 if gpu_brand == brand else 0 for brand in gpu_brand_options})
features.update({f'TypeName_{t}': 1 if type_name == t else 0 for t in type_name_options})
features.update({f'PriMemoryType_{t}': 1 if pri_memory_type == t else 0 for t in pri_memory_type_options})
features.update({f'SecMemoryType_{t}': 1 if sec_memory_type == t else 0 for t in sec_memory_type_options})

features.update({
    'Inches': inches,
    'WeightKg': weight_kg,
    'PriMemoryAmountGB': pri_memory_amount_gb,
    'SecMemoryAmountGB': sec_memory_amount_gb,
    'CpuGHz': cpu_ghz,
    'ScreenHeight': screen_height,
    'ScreenWidth': screen_width,
    'Ram(GB)': ram_gb
})

# Create a DataFrame from the features dictionary
input_df = pd.DataFrame([features])

# Ensure the input features are in the same order as during training
feature_order = ['Company_Xiaomi', 'Company_Asus', 'OpSys_Mac OS X', 'Company_Huawei', 'OpSys_Windows 10 S', 'GpuBrand_ARM', 'CpuBrand_Samsung', 'SecMemoryType_SSD', 'Company_Fujitsu', 'Company_Lenovo', 'Company_Samsung', 'Company_Google', 'OpSys_Android', 'PriMemoryType_Hybrid', 'Company_Toshiba', 'Company_HP', 'Company_Microsoft', 'Company_Dell', 'Company_Chuwi', 'Company_LG', 'Inches', 'Company_Vero', 'TypeName_2 in 1 Convertible', 'Company_Apple', 'Company_Mediacom', 'OpSys_macOS', 'TypeName_Netbook', 'SecMemoryType_Hybrid', 'OpSys_Chrome OS', 'PriMemoryAmountGB', 'OpSys_Windows 10', 'OpSys_Windows 7', 'OpSys_Linux', 'OpSys_No OS', 'Company_MSI', 'CpuBrand_AMD', 'CpuBrand_Intel', 'GpuBrand_Intel', 'GpuBrand_AMD', 'Company_Acer', 'WeightKg', 'PriMemoryType_Flash', 'Company_Razer', 'TypeName_Workstation', 'TypeName_Ultrabook', 'SecMemoryAmountGB', 'SecMemoryType_HDD', 'GpuBrand_Nvidia', 'TypeName_Gaming', 'PriMemoryType_HDD', 'CpuGHz', 'PriMemoryType_SSD', 'TypeName_Notebook', 'ScreenHeight', 'ScreenWidth', 'Ram(GB)']

# Ensure all features are present, set to 0 if missing
for feature in feature_order:
    if feature not in input_df.columns:
        input_df[feature] = 0

input_df = input_df[feature_order]

# Make prediction when user clicks the button
if st.button('Predict Price'):
    # Scale the input data
    scaled_input = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    st.subheader('Predicted Price:')
    st.write(f"€{prediction[0]:,.2f}")

# Display the input data (if selected)
if st.checkbox('Show input data'):
    st.write(input_df)

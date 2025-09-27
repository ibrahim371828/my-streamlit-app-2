#import libs
import streamlit as st
import os
import pandas as pd
import joblib
import datetime
import google.generativeai as genai
import base64
import requests
import gdown
# =========================
# Configure Gemini
# =========================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# =========================
# Extract Function
# =========================
def extract_car_details(file):
    image_bytes = file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
    You are an AI assistant that extracts structured car details from screenshots.
    Always respond ONLY in this format (no extra text):

    - make: 
    - model: 
    - variant: 
    - year: 
    - location: 
    - engine_cc: 
    - km_driven: 
    - fuel: 
    - transmission: 
    """

    response = gemini_model.generate_content(
        [
            {"role": "user", "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_base64}}
            ]}
        ]
    )

    raw_output = response.text.strip().split("\n")
    car_data = {}

    for line in raw_output:
        if ": " in line:
            key, value = line.split(": ", 1)
            key = key.strip().lower().lstrip("- ")
            value = value.strip()

            # Safely handle numeric fields with defaults
            if key == "year":
                if value.isdigit():
                    value = int(value)
                else:
                    value = 2018

            elif key == "engine_cc":
                tokens = value.split()
                value = int(tokens[0]) if tokens and tokens[0].isdigit() else 1300

            elif key == "km_driven":
                tokens = value.replace(",", "").split()
                value = int(tokens[0]) if tokens and tokens[0].isdigit() else 50000

            car_data[key] = value

    return car_data


# ===============================
# Load Model + Data
# ===============================

# Google Drive file ID
# https://drive.google.com/file/d/1pplfi-cvQDXG6-rtJTyFsPrdBnOZkOZC/view?usp=drive_link
FILE_ID = "1pplfi-cvQDXG6-rtJTyFsPrdBnOZkOZC"
MODEL_PATH = "model2.pkl"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Load model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model_rf = load_model()


# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

df_original = load_data()


# ===============================
# Streamlit App
# ===============================
st.title("🚗 Car Price Prediction App")

if "car_data" not in st.session_state:
    st.session_state.car_data = {}

# ===============================
# Capture or Upload
# ===============================
st.subheader("📸 Capture or Upload Car Screenshot")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Screenshot", type=["png", "jpg", "jpeg"])

with col2:
    captured_file = st.camera_input("Take Picture")

# Ensure save folder exists
os.makedirs("Captured_Images", exist_ok=True)

# --- Process Captured File ---
if captured_file is not None:
    img_path = os.path.join("Captured_Images", f"car_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    with open(img_path, "wb") as f:
        f.write(captured_file.getbuffer())

    with open(img_path, "rb") as f:
        st.session_state.car_data = extract_car_details(f)

    st.success("✅ Image captured, saved & processed!")

# --- Process Uploaded File ---
elif uploaded_file is not None and st.button("📥 Extract from Uploaded"):
    st.session_state.car_data = extract_car_details(uploaded_file)
    st.success("✅ Uploaded image processed!")

car_data = st.session_state.car_data

# ===============================
# Normalization Helpers
# ===============================
def normalize_make_model_variant(value):
    return str(value).strip().lower()

def normalize_location(value):
    return str(value).strip().title()

# ===============================
# Prefilled Inputs
# ===============================
# --- Select Make ---
all_makes = sorted(df_original['Make'].unique())
make = st.selectbox(
    "Car Make",
    all_makes,
    index=all_makes.index(normalize_make_model_variant(car_data.get("make", "")))
    if normalize_make_model_variant(car_data.get("make", "")) in all_makes else 0
)

# --- Filter Models based on selected Make ---
models_for_make = sorted(df_original[df_original['Make'] == make]['Model'].unique())
model_name = st.selectbox(
    "Car Model",
    models_for_make,
    index=models_for_make.index(normalize_make_model_variant(car_data.get("model", "")))
    if normalize_make_model_variant(car_data.get("model", "")) in models_for_make else 0
)

# --- Filter Variants based on selected Model ---
variants_for_model = sorted(df_original[df_original['Model'] == model_name]['Variant'].unique())
variant = st.selectbox(
    "Car Variant",
    variants_for_model,
    index=variants_for_model.index(normalize_make_model_variant(car_data.get("variant", "")))
    if normalize_make_model_variant(car_data.get("variant", "")) in variants_for_model else 0
)

all_locations = sorted(df_original['Location'].unique())
location = st.selectbox(
    "Location",
    all_locations,
    index=all_locations.index(normalize_location(car_data.get("location", "")))
    if normalize_location(car_data.get("location", "")) in all_locations else 0
)

year = st.number_input(
    "Year of Manufacture",
    min_value=1980,
    max_value=datetime.datetime.now().year,
    step=1,
    value=int(car_data.get("year", 2018)) if str(car_data.get("year", "")).isdigit() else 2018
)

engine = st.number_input(
    "Engine CC",
    min_value=660,
    max_value=4600,
    step=100,
    value=int(car_data.get("engine_cc", 1300)) if str(car_data.get("engine_cc", "")).isdigit() else 1300
)

km_driven = st.number_input(
    "KM Driven",
    min_value=0,
    max_value=1000000,
    step=1000,
    value=int(car_data.get("km_driven", 50000)) if str(car_data.get("km_driven", "")).replace(",", "").isdigit() else 50000
)

fuel_types = sorted(df_original['Fuel'].dropna().unique())
fuel = st.selectbox(
    "Fuel Type",
    fuel_types,
    index=fuel_types.index(car_data.get("fuel", "Petrol"))
    if car_data.get("fuel", "Petrol") in fuel_types else 0
)

transmission_types = sorted(df_original['Transmission'].dropna().unique())
transmission = st.selectbox(
    "Transmission",
    transmission_types,
    index=transmission_types.index(car_data.get("transmission", "Manual"))
    if car_data.get("transmission", "Manual") in transmission_types else 0
)

# ===============================
# Prediction
# ===============================
car_age = datetime.datetime.now().year - year

try:
    make_encoded = df_original.groupby('Make')['Price_in_lacs'].transform('mean')[df_original['Make'] == make].iloc[0]
    model_encoded = df_original.groupby('Model')['Price_in_lacs'].transform('mean')[df_original['Model'] == model_name].iloc[0]
    location_encoded = df_original.groupby('Location')['Price_in_lacs'].transform('mean')[df_original['Location'] == location].iloc[0]
    variant_encoded = df_original.groupby('Variant')['Price_in_lacs'].transform('mean')[df_original['Variant'] == variant].iloc[0]
except IndexError:
    st.error("⚠️ Selected option not found in training dataset.")
    st.stop()

sample = pd.DataFrame([{
    'Make_encoded': make_encoded,
    'Model_encoded': model_encoded,
    'Location_encoded': location_encoded,
    'variant_encoded': variant_encoded,
    'Year': year,
    'Engine': engine,
    'KM_driven': km_driven,
    'Car_Age': car_age,
    'Fuel': fuel,
    'Transmission': transmission
}])

if st.button("🔮 Predict Price"):
    pred_price = model_rf.predict(sample)[0]
    st.success(f"✅ Estimated Price: {pred_price:.2f} lacs")








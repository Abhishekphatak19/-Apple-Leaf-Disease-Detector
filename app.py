import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import base64

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🍎 Apple Leaf Disease Detector",
    layout="wide",
    page_icon="🍏"
)

# -------------------------------
# Model & App Data
# -------------------------------
@st.cache_resource
def load_keras_model():
    try:
        MODEL_PATH = "apple_disease_classifier_model.h5"  # ✅ best practice
        model = load_model(MODEL_PATH)  # ✅ load model here
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_keras_model()

CLASS_NAMES = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']
DISEASE_ICONS = {
    'Apple Scab': '🍏', 'Black Rot': '🖤', 'Cedar Apple Rust': '🌿', 'Healthy': '✅'
}

# -------------------------------
# LLM Initialization
# -------------------------------
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    llm = ChatOpenAI(
        openai_api_key=OPENROUTER_API_KEY,
        model="meta-llama/llama-3.1-8b-instruct",
        base_url="https://openrouter.ai/api/v1"
    )
except Exception as e:
    st.error(f"Could not initialize the LLM. Please check your API keys. Error: {e}")
    llm = None

# -------------------------------
# Custom CSS
# -------------------------------
def add_bg_from_local():
    st.markdown(
        """
        <style>

        /* Clean Dark Gradient Background */
        .stApp {
            background: radial-gradient(circle at top, #0a0f1c, #020409);
            background-attachment: fixed;
        }

        /* Main Container - Glass Effect */
        .main-container {
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            border: 1px solid rgba(0,255,170,0.15);
            box-shadow: 0 0 30px rgba(0,255,170,0.08);
        }

        /* Header */
        .card-header {
            background: rgba(0, 0, 0, 0.6);
            border-radius: 15px 15px 0 0;
            padding: 1.5rem;
            text-align: center;
            border-bottom: 1px solid rgba(0,255,170,0.2);
        }

        .card-header h1 {
            color: #00FFA6;
            font-size: 2.3rem;
            font-weight: 700;
            text-shadow: 0 0 12px rgba(0,255,170,0.6);
        }

        .card-header p {
            color: #DADADA;
            font-size: 1.1rem;
        }

        /* Content */
        .card-content {
            padding: 1.5rem;
        }

        /* Result Card */
        .result-card {
            background: rgba(0, 0, 0, 0.6);
            border-radius: 15px;
            margin-top: 2rem;
            border: 1px solid rgba(0,255,170,0.2);
            box-shadow: 0 0 20px rgba(0,255,170,0.08);
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(0, 0, 0, 0.9);
            border-right: 1px solid rgba(0,255,170,0.1);
        }

        /* Text Colors */
        h1, h2, h3, h4 {
            color: #00FFA6 !important;
        }

        p, label, span {
            color: #F5F5F5 !important;
        }

        /* Success Box */
        .stAlert {
            background: rgba(0,255,170,0.1);
            border: 1px solid rgba(0,255,170,0.3);
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #00FFA6, #00C8FF);
            color: black;
            border-radius: 10px;
            border: none;
            padding: 0.5rem 1.2rem;
            font-weight: bold;
            box-shadow: 0 0 10px rgba(0,255,170,0.4);
            transition: 0.3s;
        }

        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(0,255,170,0.7);
        }

        /* Progress Bar */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #00FFA6, #00C8FF);
        }

        /* File uploader */
        .stFileUploader {
            border: 2px dashed rgba(0,255,170,0.4);
            border-radius: 12px;
            padding: 1rem;
            background: rgba(0,0,0,0.3);
        }

        /* Image styling */
        img {
            border-radius: 10px;
            box-shadow: 0 0 15px rgba(0,255,170,0.2);
        }

        </style>
        """,
        unsafe_allow_html=True
    )
    
# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.title("📘 About This App")
    st.markdown("Welcome to the **Apple Leaf Disease Detector**.")
    st.markdown("### 🚀 How It Works")
    st.markdown("""
    1.  Upload a clear image of an **apple leaf**.
    2.  The AI analyzes it and predicts the disease.
    3.  You’ll get the result and **AI-generated care advice**.
    """)
    st.markdown("### ⚠️ Disclaimer")
    st.warning("This tool is for educational purposes. For a definitive diagnosis, consult an agricultural expert.")
    st.info("**Tech Stack:** Streamlit, TensorFlow, LangChain")

# -------------------------------
# Main Content Area
# -------------------------------
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- Main Title Header (Custom HTML) ---
st.markdown("""
    <div class="card-header main-header">
        <h1>🍎 Apple Leaf Disease Detector</h1>
        <p>Snap a picture of a plant leaf for an AI-powered diagnosis and care advice.</p>
    </div>
""", unsafe_allow_html=True)

# --- Upload section (inside a custom div for padding) ---
st.markdown('<div class="card-content">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if model and uploaded_file:
    # --- Prediction Output Card (Custom HTML) ---
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><h3>Diagnosis Result 🧠</h3></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card-content">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.image(uploaded_file, caption="Uploaded Leaf", use_container_width=True)
    with col2:
        image = Image.open(uploaded_file).convert("RGB")
        img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner('Analyzing the leaf...'):
            preds = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds)
        
        st.success(f"**{DISEASE_ICONS[predicted_class]} {predicted_class}**")
        st.write("Confidence:")
        st.progress(float(confidence))
        st.caption(f"The model is {confidence * 100:.2f}% confident.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- AI Advice Section (Now using a native Streamlit container) ---
    if llm:
        st.write("") # Add a little space
        with st.container(border=True):
            st.subheader("👩‍⚕️ AI Insight & Care Advice")
            with st.spinner("Generating expert care advice..."):
                prompt_template = ChatPromptTemplate.from_template("""
                You are an expert agricultural assistant. Provide clear, concise, and actionable advice.
                Based on the diagnosis of **{disease}**:
                1.  **Summary:** Briefly describe the disease.
                2.  **Remedies:** Suggest 2 practical treatment steps.

                Use simple language and markdown formatting (lists and bolding).
                """)
                final_prompt = prompt_template.format_messages(disease=predicted_class)
                response = llm.invoke(final_prompt)
                st.markdown(response.content, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Closes the card-content div
st.markdown('</div>', unsafe_allow_html=True) # Closes the main-container div
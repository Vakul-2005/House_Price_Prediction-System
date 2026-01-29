import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import plotly.express as px

# ---------------- BACKGROUND IMAGE ----------------
def add_bg(image_path):
    with open(image_path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #2c2c2c;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2.5rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }}
    h1,h2,h3{{color:#1f2937;font-weight:700;}}
    p,label{{color:#374151;font-size:16px;}}
    [data-testid="stSidebar"]{{background: linear-gradient(180deg, #111827,#1f2937); color:white;}}
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] p,[data-testid="stSidebar"] label {{ color: #f9fafb; }}
    .stSlider > div > div > div > div {{ background-color: #10b981; }}
    .stAlert{{background-color:#ecfdf5;color:#065f46;border-radius:12px;font-weight:600;}}
    button{{background:linear-gradient(135deg,#10b981,#059669)!important;color:white!important;border-radius:14px!important;font-size:17px!important;font-weight:700!important;padding:12px 26px!important;transition:all 0.35s ease!important;box-shadow:0 10px 25px rgba(16,185,129,0.4)!important;}}
    button:hover{{transform:translateY(-3px) scale(1.04);box-shadow:0 18px 35px rgba(16,185,129,0.6);}}
    button:active{{transform:scale(0.96);}}
    </style>
    """, unsafe_allow_html=True)

add_bg("assets/background.png")

# ---------------- LOAD MODEL & DATA ----------------
model = joblib.load("models/house_price_model.pkl")
data = pd.read_csv("data/house_data.csv")

st.set_page_config(page_title="AI House Price Prediction", layout="wide")

# ---------------- TITLE ----------------
st.title("🏠 AI House Price Prediction System")
st.markdown("### Predict house prices using Machine Learning / AI")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🏡 Enter House Details")
area = st.sidebar.slider("Area (sq ft)", 500, 5000, 1500)
bedrooms = st.sidebar.slider("Bedrooms", 1, 6, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1, 5, 2)
stories = st.sidebar.slider("Stories", 1, 4, 1)
parking = st.sidebar.slider("Parking Spaces", 0, 4, 1)

# ---------------- PREDICTION BUTTON ----------------
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔮 Predict Price"):
    input_data = pd.DataFrame(
        [[area, bedrooms, bathrooms, stories, parking]],
        columns=['area','bedrooms','bathrooms','stories','parking']
    )
    prediction = model.predict(input_data)[0]
    st.subheader("💰 Predicted House Price")
    st.success(f"₹ {prediction:,.0f}")

# ---------------- ANIMATED / INTERACTIVE CHARTS ----------------
st.subheader("📊 Data Insights")

# 1️⃣ Price Distribution
fig1 = px.histogram(data, x="price", nbins=20, color_discrete_sequence=["#10b981"], title="Price Distribution")
fig1.update_layout(xaxis_title="Price (₹)", yaxis_title="Count", bargap=0.2)
st.plotly_chart(fig1, width='stretch')   # full width



# 2️⃣ Area vs Price Scatter
fig2 = px.scatter(
    data, x="area", y="price", size="bedrooms", color="bathrooms",
    hover_data=["stories","parking"], title="Area vs Price", color_continuous_scale=px.colors.sequential.Teal
)
st.plotly_chart(fig2, width='stretch')   # full width scatter



# 3️⃣ Feature Correlation Heatmap
corr = data.corr()
fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="YlGnBu", title="Feature Correlation")
st.plotly_chart(fig3, width='stretch')   # heatmap


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("👨‍💻 Built with Python, AI/ML, Streamlit & Plotly")

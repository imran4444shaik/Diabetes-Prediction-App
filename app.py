import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")
    return df

df = load_data()

# Train model
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🧠 Diabetes Prediction App")

st.sidebar.header("Input Features")
pregnancies = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 0, 140, 70)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 10, 100, 30)

input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]
proba = model.predict_proba(input_scaled)[0][1]

st.subheader("Prediction Result")
if prediction == 1:
    st.error(f"⚠️ High risk of diabetes ({proba:.2%} probability)")
else:
    st.success(f"✅ Low risk of diabetes ({proba:.2%} probability)")

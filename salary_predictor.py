import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Salary Predictor Pro",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(to right, #6e8efb, #a777e3);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #5e7ceb, #9667d3);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sample data
@st.cache_data
def load_data():
    data = {
        'Age': np.random.randint(22, 60, 500),
        'YearsExperience': np.random.uniform(1, 20, 500).round(1),
        'EducationLevel': np.random.choice(["Bachelor's", "Master's", "PhD"], 500),
        'Department': np.random.choice(["Engineering", "Marketing", "HR", "Finance", "Operations"], 500),
        'Gender': np.random.choice(["Male", "Female"], 500),
        'Salary': np.random.uniform(30000, 150000, 500).round(2)
    }
    return pd.DataFrame(data)

df = load_data()

# Header
st.title("💰 Employee Salary Prediction")
st.markdown("Predict employee salaries based on key attributes")

# Sidebar for user inputs
with st.sidebar:
    st.header("🧑‍💼 Employee Details")
    
    age = st.slider("Age", 22, 60, 30)
    experience = st.slider("Years of Experience", 1.0, 20.0, 5.0, step=0.5)
    education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
    department = st.selectbox("Department", ["Engineering", "Marketing", "HR", "Finance", "Operations"])
    gender = st.radio("Gender", ["Male", "Female"])
    
    predict_button = st.button("Predict Salary")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Data visualization
    st.header("📊 Salary Distribution")
    
    # Interactive filters for visualization
    vis_filter = st.selectbox(
        "View salary distribution by:",
        ["Department", "Education Level", "Gender"]
    )
    
    if vis_filter == "Department":
        fig = px.box(df, x='Department', y='Salary', color='Department')
    elif vis_filter == "Education Level":
        fig = px.box(df, x='EducationLevel', y='Salary', color='EducationLevel')
    else:
        fig = px.box(df, x='Gender', y='Salary', color='Gender')
        
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.header("🧠 Prediction Model")
    
    # Prepare data for model
    df['EducationLevel'] = df['EducationLevel'].map({"Bachelor's": 0, "Master's": 1, "PhD": 2})
    df['Department'] = df['Department'].astype('category').cat.codes
    df['Gender'] = df['Gender'].map({"Male": 0, "Female": 1})
    
    X = df[['Age', 'YearsExperience', 'EducationLevel', 'Department', 'Gender']]
    y = df['Salary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Show model metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.metric("Mean Squared Error", f"${mse:,.2f}")
    
    # Prediction
    if predict_button:
        try:
            input_data = pd.DataFrame({
                'Age': [age],
                'YearsExperience': [experience],
                'EducationLevel': [["Bachelor's", "Master's", "PhD"].index(education)],
                'Department': [[0, 1, 2, 3, 4][["Engineering", "Marketing", "HR", "Finance", "Operations"].index(department)]],
                'Gender': [0 if gender == "Male" else 1]
            })
            
            prediction = model.predict(input_data)
            st.success(f"Predicted Salary: **${prediction[0]:,.2f}**")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Data table
st.header("📋 Employee Data Sample")
st.dataframe(df.sample(10), use_container_width=True)

st.markdown("---")
st.caption("Note: This application uses synthetic data for demonstration purposes.")

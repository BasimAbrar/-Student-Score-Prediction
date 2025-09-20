import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_resource
def train_and_save_model():
    df = pd.read_csv("StudentPerformanceFactors.csv")

    df = df.drop_duplicates()
    df = df.dropna()

    df = df[(df["Exam_Score"] >= 0) & (df["Exam_Score"] <= 100)]

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    X = df_encoded.drop("Exam_Score", axis=1)
    y = df_encoded["Exam_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Model trained → MSE:", mse, "R2:", r2)

    joblib.dump((model, list(X.columns)), "LinearRegression_FullModel.joblib")

    return model, list(X.columns)


try:
    model, feature_names = joblib.load("LinearRegression_FullModel.joblib")
except:
    model, feature_names = train_and_save_model()


st.set_page_config(page_title="Exam Score Predictor", page_icon="🎓", layout="centered")

st.title(" Student Exam Score Predictor")
st.write("This app predicts a student's exam score based on study and lifestyle factors.")


st.sidebar.header("Enter Student Details")

hours_studied = st.sidebar.number_input("Hours Studied", min_value=0, max_value=24, value=5)
attendance = st.sidebar.slider("Attendance (%)", min_value=0, max_value=100, value=80)
sleep_hours = st.sidebar.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
previous_scores = st.sidebar.number_input("Previous Scores", min_value=0, max_value=100, value=70)
tutoring_sessions = st.sidebar.number_input("Tutoring Sessions", min_value=0, max_value=10, value=2)

parental_involvement = st.sidebar.selectbox("Parental Involvement", ["Low", "Medium", "High"])
access_to_resources = st.sidebar.selectbox("Access to Resources", ["Low", "Medium", "High"])
internet_access = st.sidebar.radio("Internet Access", ["No", "Yes"])
teacher_quality = st.sidebar.selectbox("Teacher Quality", ["Low", "Medium", "High"])
school_type = st.sidebar.radio("School Type", ["Private", "Public"])


input_dict = {
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores,
    "Tutoring_Sessions": tutoring_sessions,
    "Parental_Involvement_Low": 1 if parental_involvement == "Low" else 0,
    "Parental_Involvement_Medium": 1 if parental_involvement == "Medium" else 0,
    "Parental_Involvement_High": 1 if parental_involvement == "High" else 0,
    "Access_to_Resources_Low": 1 if access_to_resources == "Low" else 0,
    "Access_to_Resources_Medium": 1 if access_to_resources == "Medium" else 0,
    "Access_to_Resources_High": 1 if access_to_resources == "High" else 0,
    "Internet_Access_No": 1 if internet_access == "No" else 0,
    "Internet_Access_Yes": 1 if internet_access == "Yes" else 0,
    "Teacher_Quality_Low": 1 if teacher_quality == "Low" else 0,
    "Teacher_Quality_Medium": 1 if teacher_quality == "Medium" else 0,
    "Teacher_Quality_High": 1 if teacher_quality == "High" else 0,
    "School_Type_Private": 1 if school_type == "Private" else 0,
    "School_Type_Public": 1 if school_type == "Public" else 0,
}

input_data = pd.DataFrame([[input_dict.get(col, 0) for col in feature_names]], columns=feature_names)


if st.button("Predict Exam Score"):
    prediction = model.predict(input_data)[0]
    st.success(f" Predicted Exam Score: **{prediction:.2f}** / 100")

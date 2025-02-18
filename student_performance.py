import numpy as np
import joblib
import streamlit as st

# Load the trained model
model = joblib.load('trained_Student_Regressor_Model.pkl')

st.title("Student Performance Predictor")

st.header("Enter the Student Data:")

hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=168.0, step=0.1,help="Number of hours spent studying per week.")
attendance = st.slider("Attendance (%)", min_value=0, max_value=100, step=1, help="Percentage of classes attended.")
tutoring_sessions = st.number_input("Number of Tutoring Sessions", min_value=0, max_value=100, step=1, help="Number of tutoring sessions attended per month.")
physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0, max_value=24, step=1, help="Average number of hours of physical activity per week.")

parental_involvement = st.radio("Parental Involvement:", ["High", "Low", "Medium"],help="Level of parental involvement in the student's education.")
access_to_resources = st.radio("Access to Resources:", ["High", "Low", "Medium"],help="Availability of educational resources")
motivation_level = st.radio("Motivation Level:", ["High", "Low", "Medium"],help="Student's level of motivation.")
internet_access = st.radio("Internet Access:", ["Yes", "No"],help="Availability of internet access.")
family_income = st.radio("Family Income:", ["High", "Low", "Medium"],help="Family income level.")
teacher_quality = st.radio("Teacher Quality:", ["High", "Low", "Medium"],help="Quality of teachers.")
school_type = st.radio("School Type:", ["Private", "Public"],help="Type of school attended.")
peer_influence = st.radio("Peer Influence:", ["Positive", "Negative", "Neutral"],help="Influence of peers on academic performance.")
learning_disabilities = st.radio("Learning Disabilities:", ["Yes", "No"],help="Presence of learning disabilities.")
parental_education_level = st.selectbox(
    "Parental Education Level:",
    ["College", "High School", "Postgraduate"],help="Highest education level of parents."
)
distance_from_home = st.radio("Distance from Home:", ["Far", "Near", "Moderate"],help="Distance from home to school.")

# Maps categorical data to the columns. If medium is selected, the if statements will not trigger and both columns = 0
inputs = [
    hours_studied,
    attendance,
    tutoring_sessions,
    physical_activity,

    1 if parental_involvement == "High" else 0,
    1 if parental_involvement == "Low" else 0,

    1 if access_to_resources == "High" else 0,
    1 if access_to_resources == "Low" else 0,

    1 if motivation_level == "High" else 0,
    1 if motivation_level == "Low" else 0,

    1 if internet_access == "Yes" else 0,

    1 if family_income == "High" else 0,
    1 if family_income == "Low" else 0,

    1 if teacher_quality == "High" else 0,
    1 if teacher_quality == "Low" else 0,

    1 if school_type == "Private" else 0,

    1 if peer_influence == "Negative" else 0,
    1 if peer_influence == "Positive" else 0,

    1 if learning_disabilities == "Yes" else 0,

    1 if parental_education_level == "College" else 0,
    1 if parental_education_level == "High School" else 0,
    1 if parental_education_level == "Postgraduate" else 0,

    1 if distance_from_home == "Far" else 0,
    1 if distance_from_home == "Near" else 0,
]

# Prediction
if st.button("Predict Performance"):
    try:
        features = np.array([inputs])
        prediction = model.predict(features)

        # Force prediction within range 0-100
        prediction = np.clip(prediction, 0, 100)

        # Determine the range for the prediction
        if prediction <= 10:
            result_range = "0-10"
            face_image = "images/very_sad_face.jpg"
        elif prediction <= 20:
            result_range = "10-20"
            face_image = "images/very_sad_face.jpg"
        elif prediction <= 30:
            result_range = "20-30"
            face_image = "images/sad_face.jpg"
        elif prediction <= 40:
            result_range = "30-40"
            face_image = "images/sad_face.jpg"
        elif prediction <= 50:
            result_range = "40-50"
            face_image = "images/neutral_face.jpg"
        elif prediction <= 60:
            result_range = "50-60"
            face_image = "images/neutral_face.jpg"
        elif prediction <= 70:
            result_range = "60-70"
            face_image = "images/happy_face.jpg"
        elif prediction <= 80:
            result_range = "70-80"
            face_image = "images/happy_face.jpg"
        elif prediction <= 90:
            result_range = "80-90"
            face_image = "images/very_happy_face.jpg"
        else:
            result_range = "90-100"
            face_image = "images/very_happy_face.jpg"

        st.success(f"The predicted student performance is in the range: {result_range} / 100")

        st.image(face_image, caption=f"Remember, it's just an estimate", width=100)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

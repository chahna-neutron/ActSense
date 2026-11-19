import random
import streamlit as st
import pandas as pd

st.set_page_config(page_title="ActSense AI", layout="wide")
st.title("ActSense AI - Human Activity Recognition")

ACTIVITIES = ["walking", "running", "sitting", "standing", "jumping"]

if "history" not in st.session_state:
    st.session_state.history = []


def get_prediction():
    # DUMMY: later replace with Member 1 + Member 2 code
    activity = random.choice(ACTIVITIES)
    confidence = round(random.uniform(70, 99), 1)
    return activity, confidence


left, right = st.columns(2)

with left:
    st.subheader("Webcam")
    st.camera_input("Camera")

with right:
    st.subheader("Prediction")
    if st.button("Predict Activity"):
        activity, confidence = get_prediction()
        st.session_state.history.append(
            {"Activity": activity, "Confidence (%)": confidence}
        )

    if st.session_state.history:
        last = st.session_state.history[-1]
        st.metric("Activity", last["Activity"].upper())
        st.metric("Confidence", f"{last['Confidence (%)']}%")
        st.progress(int(last["Confidence (%)"]))

st.divider()

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Activity History")
        st.dataframe(df, use_container_width=True)
    with c2:
        st.subheader("Activity Graph")
        st.bar_chart(df["Activity"].value_counts())

    st.subheader("Confidence Over Time")
    st.line_chart(df["Confidence (%)"])
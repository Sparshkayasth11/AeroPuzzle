import streamlit as st

st.set_page_config(page_title="AeroPuzzle", layout="centered")

st.title("AeroPuzzle")
st.write(
    "This repository currently contains a desktop webcam-based game, not a full Streamlit web app."
)
st.write(
    "Render is configured to launch `streamlit_app.py`. This placeholder page allows deployment to start successfully."
)

st.markdown(
    "### Run locally\n" 
    "Install dependencies and use `python -m aeropuzzle` to launch the desktop app with webcam support."
)

st.code("python -m aeropuzzle")

st.warning(
    "The actual game uses OpenCV and webcam access, which may not work on Render's hosted environment. "
    "This page only resolves the missing file error."
)

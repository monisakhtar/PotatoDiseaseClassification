import streamlit as st
import requests
from PIL import Image

# st.set_page_config(layout="wide")


# Define the FastAPI endpoint URL
API_URL = 'http://localhost:8000/uploadfile/'



page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://t.ly/9_s4e");
background-size: 100%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}


[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Page title and header

st.markdown(
        """
        <div style="margin-bottom: 50px;">
            <span style="font-size: 56px; font-weight: bold; color: #f5f5dc;">Potato Disease Classification</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
st.markdown("</div>", unsafe_allow_html=True)


containter = st.container()
with containter:


    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["JPG", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        st.write("")
        # Read the file content as bytes
        file_bytes = uploaded_file.read()

        # Send the image file to FastAPI backend
        files = {"image": ("image.jpg", file_bytes, "image/jpeg")}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            prob = result['Probability'] * 100

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                        """
                        <div style="margin-bottom: 5px;">
                            <span style="font-size: 26px; font-weight: bold; color: black;">Prediction</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
                # st.write("Prediction")
            with col2:
                st.markdown(
                        """
                        <div style="margin-bottom: 5px;">
                            <span style="font-size: 26px; font-weight: bold; color: black;">Probability</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 50px;">
                        <span style="font-size: 26px; font-weight: bold; color: black;">{result['Prediction']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
                # st.write(f"{result['Prediction']}")
            with col2:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 50px;">
                        <span style="font-size: 26px; font-weight: bold; color: black;">{prob:.2f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("An error occurred while processing the image.")
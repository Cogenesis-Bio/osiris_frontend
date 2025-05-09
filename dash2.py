import streamlit as st
import requests

st.title("Osiris: HSC Fate + Lineage Bias Predictor")

st.write("Upload a `.tsv` file and we'll tell you the HSC fate and lineage bias.")

uploaded_file = st.file_uploader("Upload your `.tsv` file here", type=["tsv"])

# Replace IP since we're not running it locally
backend_url = "http://172.22.163.105:8080/predict"

if uploaded_file is not None:
    if st.button("Run Prediction"):
        try:
            # Send the file as multipart/form-data
            files = {"file": uploaded_file}
            response = requests.post(backend_url, files=files)

            if response.status_code == 200:
                data = response.json()
                
                st.success(data["message"])

                # Show HSC predictions
                st.subheader("HSC Fate Prediction")
                for result in data.get("hsc_predictions", []):
                    st.write(f"Class: **{result['class']}**")
                    st.write(f"Probability: **{result['probability']}**")

                # Show Lineage predictions
                st.subheader("Lineage Bias Prediction")
                for result in data.get("lineage_predictions", []):
                    st.write(f"Class: **{result['class']}**")
                    st.write(f"Probability: **{result['probability']}**")

            else:
                st.error(f"Server returned an error: {response.status_code}")
                st.json(response.json())

        except Exception as e:
            st.error(f"Could not connect to backend: {e}")

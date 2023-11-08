import streamlit as st
from fastai.vision.all import *
import urllib.request

st.write("Cat vs. Dog Classifier")
st.text("Built by Jayden Chong")


def label_function(f): return f[0].isupper()


model = load_learner('my_model.pkl')


def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = model.predict(img)
    likelihood_is_cat = outputs[1].item()
    if likelihood_is_cat > 0.9:
        return "Cat"
    elif likelihood_is_cat < 0.1:
        return "Dog"
    else:
        return "Not sure… try another picture!"


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
if st.button("Predict"):
    prediction = predict(uploaded_file)
    st.write(prediction)

st.text("Built with Streamlit and Fastai")

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Cat vs Dog Predictor", page_icon="🐾")
# Streamlit ke default Header (Fork/GitHub) aur Footer ko hide karne ka CSS
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.title("🐱 vs 🐶 Cat & Dog Classifier")

# Saved Model ko Load karna
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('cat_dog_model.h5')

try:
    cnn = load_my_model()
except Exception as e:
    st.error("⚠️ 'cat_dog_model.h5' File not found! First, run `cnn.save('cat_dog_model.h5')` in the notebook to save the model.")
    st.stop()

# Frontend se Image Upload karna
uploaded_file = st.file_uploader("Please choose Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Frontend par Uploaded Image dikhana
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    if st.button("Predict"):
        # Image processing
        test_image = image.load_img(uploaded_file, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        
        # Scaling karna bohot zaroori hai
        test_image = test_image / 255.0 
        
        test_image = np.expand_dims(test_image, axis=0)
        
        # Prediction lena
        result = cnn.predict(test_image)
        confidence = result[0][0] # 0 se 1 ke beech value aayegi
        
        # --- Naya Threshold Logic ---
        # 1. Agar value 0.80 ya us se zyada hai
        if confidence >= 0.80:
            st.success(f"🐶 *Prediction:* Dog \n\nConfidence: {confidence * 100:.2f}%")
            
        # 2. Agar value 0.20 ya us se kam hai
        elif confidence <= 0.20:
            cat_confidence = (1 - confidence) * 100
            st.success(f"🐱 *Prediction:* Cat \n\nConfidence: {cat_confidence:.2f}%")
            
        # 3. Agar value 0.20 aur 0.80 ke beech mein aati hai
        else:
            st.error("⚠️ *ERROR:* This image does not clearly look like a cat or a dog. The model is confused, so please upload the correct image!")
    
    # if st.button("Predict"):
    #     # 2. AAPKA NOTEBOOK WALA CODE (MODIFIED FOR STREAMLIT)
    #     # Path ki jagah directly uploaded_file pass karenge
    #     test_image = image.load_img(uploaded_file, target_size=(64, 64))
    #     test_image = image.img_to_array(test_image)
        
        #  Model training ke waqt rescale=1./255 kiya tha, isiliye yahan / 255.0 karna zaroori hai
    #     test_image = test_image / 255.0 
        
    #     test_image = np.expand_dims(test_image, axis=0)
        
    #     # Prediction logic
    #     result = cnn.predict(test_image)
        
    #     # Result check karna
    #     if result[0][0] > 0.5:
    #         prediction = 'dog'
    #         st.success("🐶 **Prediction:** Dog (Kutta)")
    #     else:
    #         prediction = 'cat'
    #         st.success("🐱 **Prediction:** Cat (Billi)")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

model_path = r'D:\Semester 5\Pembelajaran Mesin Dan Mendalam\module 5\Introduction to Deep Learning (Praktek)\Introduction to Deep Learning (Praktek)\best_model_tf.h5'

if os.path.exists(model_path):
    try:
        
        tf.get_logger().setLevel('ERROR')
        model = tf.keras.models.load_model(model_path, compile=False)

        
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        
        def preprocess_image(image):
            image = image.resize((28, 28))  
            image = image.convert('L')  
            image_array = np.array(image) / 255.0  
            image_array = image_array.reshape(1, 28, 28, 1)  
            return image_array

        
        st.title("Fashion MNIST Image Classifier")
        st.write("Unggah gambar item fashion (misalnya sepatu, tas, baju), dan model akan memprediksi kelasnya.")

        
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

       
        if uploaded_file is not None:
            
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

            
            if st.button("Predict"):
                
                processed_image = preprocess_image(image)
                predictions = model.predict(processed_image)[0]

               
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class] * 100  


                st.write("### Hasil Prediksi")
                st.write(f"**Kelas Prediksi: {class_names[predicted_class]}**")
                st.write(f"**Confidence: {confidence:.2f}%**")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.error("File model tidak ditemukan.")

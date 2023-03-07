#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model('dog_breed.h5')         #THis will save our whole model in "model" variable

#Name of Classes
CLASS_NAMES = ['Scottish Deerhound','Maltese Dog','Bernese Mountain Dog']          #Since we are only working on 3 breeds.

#Setting Title of App
st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

#Uploading the dog image
dog_image = st.file_uploader("Choose an image...", type="png")
submit = st.button('Predict')
#On predict button click
if submit:


    if dog_image is not None:                     #it will check the variable "dog_image" and see if it is not empty, if it is not empty then it will proceed

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)      #convert it into a numpy array 
        opencv_image = cv2.imdecode(file_bytes, 1)                                 #Convert numpy array into opencv_image



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (224,224))          #Our model take image of size 224,224 hence we are resizing the image
        #Convert image to 4 Dimension
        opencv_image.shape = (1,224,224,3)                             #Now we will change our image shape to 4 dimension
        #Make Prediction
        Y_pred = model.predict(opencv_image)                           #Now we can give our image to model for prediction

        st.title(str("The Dog Breed is "+CLASS_NAMES[np.argmax(Y_pred)]))       #To print the Dog Breed Name

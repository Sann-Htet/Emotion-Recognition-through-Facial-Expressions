import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical,img_to_array
from PIL import Image
import streamlit as st
import time
from fastai.vision import pil2tensor
from skimage import color

st.title("Welcom to Face Emotion Recognition Application")

main_options = st.selectbox("What would like to choose: ", ['About', 'Detection space', 'Contact'])

if main_options == 'About':
  st.write('This is the solution to real time face emotion detection problem')
  st.write('Today, the majority of our time is spent on interacting with computers and mobile phones in our daily life due to \
  technology progression and ubiquitously spreading these mediums. However, they play an essential role in our \
  lives, and the vast number of existing software interfaces are non-verbal, primitive and terse. Adding emotional\
  expression recognition to expect the users’ feelings and emotional state can drastically improve human–computer interaction ')
  st.write('Humans usually employ different cues to express their emotions, such as facial expressions, hand \
  gestures and voice. Facial expressions represent up to 55% of human communications while other ways such as \
  oral language are allocated a mere 7% of emotion expression. Therefore, considering facial expressions in an \
  HRI(Human robotic interactions system enables simulation of natural interactions successfully')
  st.write('This has many more advantages than we can imagine')
  st.write('This can be used in education, research, medicine, manufacturing, investigation and many more')
  st.markdown(
        '### How good it is that the people are able to catch the things or understand properly, Same with real time detection')
  st.write('This when used in right way gives many good results')
  st.write('Note: This was created with the objective of using in field of education which detect the emotions\
  of the students which enables in proper understanding and teaching the students the right way to do the things.\
  Works even like surveillance camera which keeps the eye on students emotions')

elif main_options == 'Contact':
  st.write('''
               https://www.linkedin.com/in/sann-htet-62864a21b/
                      ''')
elif main_options == 'Detection space':
  option = st.radio('Which type of detection would you like to make?',
                      ('an Image', 'a Video', 'OpenCV Live', 'an Instant Snapshot live detection',
                       'a Live Video detection'))
  st.header('You selected {} option for emotion detection'.format(option))

  if option == 'an Image':
    image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
    if image_file is not None:
      image2 = Image.open(image_file)
      #file_bytes = np.asarray(bytearray(image_file.read()),dtype=np.uint8)
      #opencv_image = cv2.imdecode(file_bytes,1)
      st.text("Original Image")
      #st.image(opencv_image,channels="BGR",use_column_width=True)
      #gray1 = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2GRAY)
      progress = st.progress(0)
      for i in range(100):
        time.sleep(0.01)
        progress.progress(i+1)
      st.image(image2)
    #if image_file in None:
      #st.error("No image uploaded yet")

    if st.button("Progress"):
        
      model = load_model('/content/gdrive/MyDrive/cnn.h5')
      #gray = cv2.cvtColor(opencv_image,cv2.COLOR_BGR2GRAY)
      #t = pil2tensor(opencv_image,dtype=np.float32)
      #t = t.float()/255.0
      #gray = gray/255.0
      #img1 = Image.open(t)
      #img = cv2.cvtColor(np.float32(image2),1)
      #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      #images = cv2.resize(gray,(48,48))
      image = np.array(image2)
      img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      roi_gray = cv2.resize(img_gray,(48,48),interpolation=cv2.INTER_AREA)
      roi = roi_gray.astype('float')/255.0
      roi = img_to_array(roi)
      roi = np.expand_dims(roi,axis=0)
      #img = image.load_img(image2,color_mode='grayscale',target_size=(48,48,1))
      #y = image.img_to_array(images)
      #y = np.expand_dims(y,axis=0)
      #images = np.vstack([y])
      #classes = model.predict(images/255.0)
      #images = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
      #images = cv2.resize(images,(48,48),interpolation=cv2.INTER_AREA)
      #images = images/255.0
      classes = model.predict(roi.reshape(-1,48,48,1))
      labels_index = ["angry","disgust","fear","happy","sad","surprise","neutral"]
      max_index = np.argmax(classes)
      prediction = labels_index[max_index]
      print("This person here is",labels_index[max_index])
      if prediction == 'Happy':
        st.subheader("YeeY!  You are Happy :smile: today Always Be ! ")
      elif prediction == 'sad':
        st.subheader("You seem to be sad today, Smile and be happy!")
      elif prediction == 'angry':
        st.subheader("You seem to be angry, Take it easy!")
      elif prediction == 'disgust':
        st.subheader("You seem to be Disgust today!")
      elif prediction == 'fear':
        st.subheader("You seem to be Fearful, Be couragous!")
      elif prediction == 'surprise':
        st.subheader("You seem to be surprise today!")
      else:
        st.subheader("You seem to be Neutral today, Happy day!")
    
  elif option == 'a Video':
    st.write('This is not a good option for us')
    st.write('We will contact you when this option is ready to use')

  elif option == 'OpenCV Live':
    st.write('This is not a good option for streamlit then it works well locally')
    st.write('OpenCV is popular python library used when images and videos are involved')
    st.write('OpenCV is a good option for computer vision problems')
    st.markdown(
            '### For real time detection in streamlit please use a Live Video detection option/an Instant Snapshot detection for finding out the emotion of a person in live')
    st.write(
        "Streamlit doesn't support OpenCV for live detection for some reasons and webrtc solves this problem in streamlit. Hence choose the other options for detection")
    st.write('Thanks for reading')
    st.write('Thank you')
  else:
    st.write('You did not select the proper option as specified. Please select a valid option')
    st.write(
          'If one of the four options was selected and it did not work. Please clear the cache and rerun the application')
    st.write('Thanks for understanding')

st.write('Thank you. I hope you got emotions detected which are hidden in the picture or an image or a video')
st.write('See you soon')
st.write('This is created by Team3Group')
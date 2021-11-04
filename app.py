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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

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
      labels_index = ["anger","disgust","fear","happiness","sadness","surprise","neutral"]
      max_index = np.argmax(classes)
      prediction = labels_index[max_index]
      print("This person here is",labels_index[max_index])
      if prediction == 'happiness':
        st.subheader("YeeY!  You are Happy :smile: today Always Be ! ")
      elif prediction == 'sadness':
        st.subheader("You seem to be sad today, Smile and be happy!")
      elif prediction == 'anger':
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


    # load model
    emotion_dict = {0:'angry', 1 :'disgust', 2: 'fear', 3:'happy', 4: 'sad',5:'surprise',6:'neutral'}
    classifier = load_model('/content/gdrive/MyDrive/cnn.h5')

    #load face
    try:
        face_cascade = cv2.CascadeClassifier('/content/gdrive/MyDrive/haarcascade_frontalface_default.xml')
    except Exception:
        st.write("Error loading cascade classifiers")

    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    class Faceemotion(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            #image gray
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                image=img_gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img=img, pt1=(x, y), pt2=(
                    x + w, y + h), color=(255, 0, 0), thickness=2)
                roi_gray = img_gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    prediction = classifier.predict(roi)[0]
                    maxindex = int(np.argmax(prediction))
                    finalout = emotion_dict[maxindex]
                    output = str(finalout)
                label_position = (x, y)
                cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            return img

    def main():
        # Face Analysis Application #
        st.title("Real Time Face Emotion Detection Application")
        activiteis = ["Home", "Webcam Face Detection", "About"]
        choice = st.sidebar.selectbox("Select Activity", activiteis)
        st.sidebar.markdown(
            """ Developed by Computer vision Team 3 group    
                Email : sannhtet899@gmailcom """)
        if choice == "Home":
            html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                                <h4 style="color:white;text-align:center;">
                                                Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                                </div>
                                                </br>"""
            st.markdown(html_temp_home1, unsafe_allow_html=True)
            st.write("""
                    The application has two functionalities.

                    1. Real time face detection using web cam feed.

                    2. Real time face emotion recognization.

                    """)
        elif choice == "Webcam Face Detection":
            st.header("Webcam Live Feed")
            st.write("Click on start to use webcam and detect your face emotion")
            webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                            video_processor_factory=Faceemotion)

        elif choice == "About":
            st.subheader("About this app")
            html_temp_about1= """<div style="background-color:#6D7B8D;padding:10px">
                                        <h4 style="color:white;text-align:center;">
                                        Real time face emotion detection application using OpenCV, Custom Trained CNN model and Streamlit.</h4>
                                        </div>
                                        </br>"""
            st.markdown(html_temp_about1, unsafe_allow_html=True)

            html_temp4 = """
                                    <div style="background-color:#98AFC7;padding:10px">
                                    <h4 style="color:white;text-align:center;">This Application is developed by Mohammad Juned Khan using Streamlit Framework, Opencv, Tensorflow and Keras library for demonstration purpose. If you're on LinkedIn and want to connect, just click on the link in sidebar and shoot me a request. If you have any suggestion or wnat to comment just write a mail at Mohammad.juned.z.khan@gmail.com. </h4>
                                    <h4 style="color:white;text-align:center;">Thanks for Visiting</h4>
                                    </div>
                                    <br></br>
                                    <br></br>"""

            st.markdown(html_temp4, unsafe_allow_html=True)

        else:
            pass


    if __name__ == "__main__":
        main()


  else:
    st.write('You did not select the proper option as specified. Please select a valid option')
    st.write(
          'If one of the four options was selected and it did not work. Please clear the cache and rerun the application')
    st.write('Thanks for understanding')

st.write('Thank you. I hope you got emotions detected which are hidden in the picture or an image or a video')
st.write('See you soon')
st.write('This is created by Computer vision Team3 group')

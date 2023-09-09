from PIL import Image
import numpy as np
import streamlit as st
import torch
import cv2
import datetime
from datetime import date
import psutil
import pandas as pd
import time
import torchvision.transforms as transforms
import tempfile
import ultralytics 



st.set_page_config(page_title="Indian Sign Language Detection", page_icon="👋")

centered_style = "text-align: center;"
# Add the "Home Screen" button to the sidebar
if st.sidebar.button("Home"):
    st.experimental_rerun()

def about_page():


    # Set the text-align CSS style to center
    

    col2, col1 = st.columns([3,1])
    # with col3:
    #     st.image("logo1.png", use_column_width="auto")

    # Profile Picture
    with col1:
        st.image("anirban.jpg", use_column_width="auto")
        st.markdown("<p style='{}'><strong>Anirban Bhattacharjee, M.Sc., M.Tech (Developer)</strong></p>".format(centered_style), unsafe_allow_html=True)
        st.markdown(" ")
        st.markdown(" ")
        st.image("anjansir.jpg", use_column_width="auto")
        st.markdown("<p style='{}'><strong>Dr. Anjan Kr. Talukdar, (Project Supervisor) Assistant Professor, Dept. of Electronics & Communication Engineering, Gauhati University</strong></p>".format(centered_style), unsafe_allow_html=True)


    # About
    with col2:
        # Header
        st.markdown("<h1 style='{}'>About Us</h1>".format(centered_style), unsafe_allow_html=True)

        # Description
        st.markdown("<h3 style='{}'>👋 Welcome to our Multi-Modal ISL Recognition System!</h1>".format(centered_style), unsafe_allow_html=True)
        # st.markdown("<p>👋 Welcome to our Multi-Modal ISL Recognition System!</p>", unsafe_allow_html=True)
        st.markdown("<p>A M.Tech project for recognizing Indian Sign Language using computer vision and deep learning.</p>", unsafe_allow_html=True)

        # Quote
        st.markdown("<blockquote><p>'In the realm of silence, the hands speak the language of the heart.'</p></blockquote>", unsafe_allow_html=True)

        st.markdown("<p>A novel approach that enables efficient and precise sign language detection is the Real-Time Continuous Indian Sign Language detection system. This system utilizes YOLOv5 with Streamlit UI to detect static motions, alphabets, numerals, and dynamic gestures in real-time within video feeds by utilizing object detection capabilities.</p>", unsafe_allow_html=True)
        st.markdown("<p>The architecture of YOLOv5 provides great recall and precision, producing outstanding recognition performance across various sign language components. Static photos, webcam captures, and live webcam feeds may all be used to seamlessly interact with the system thanks to the user-friendly Streamlit UI. In addition to addressing communication difficulties experienced by the deaf population, this initiative shows how cutting-edge technologies may promote inclusivity and equal opportunity for all.</p>", unsafe_allow_html=True)
        # st.markdown("<p>In addition to addressing communication difficulties experienced by the deaf population, this initiative shows how cutting-edge technologies may promote inclusivity and equal opportunity for all.</p>", unsafe_allow_html=True)

        st.markdown(" ")
        st.markdown(" ")
        # Contact Information
        st.markdown("<h3>Contact Information</h3>", unsafe_allow_html=True)
        st.markdown("<p>Developed By:</p>", unsafe_allow_html=True)
        st.markdown("<p>Anirban Bhattacharjee</p>", unsafe_allow_html=True)
        st.markdown("<p>Department of Electronics & Communication Engineering,</p>", unsafe_allow_html=True)
        st.markdown("<p>Gauhati University</p>", unsafe_allow_html=True)
        st.markdown("<p>781014, Guwahati, Assam</p>", unsafe_allow_html=True)
        st.markdown("<p><strong>Email:</strong> anirban8472@gmail.com</p>", unsafe_allow_html=True)
        st.markdown("<p><strong>LinkedIn:</strong> <a href='https://www.linkedin.com/in/anirban-bhattacharjee-10063a150/' target='_blank'>Anirban Bhattacharjee</a></p>", unsafe_allow_html=True)

        # Footer
        st.markdown("<p style='{}'>Thank you for visiting!</p>".format(centered_style), unsafe_allow_html=True)



if st.sidebar.button("About Us"):
    # Clear the current content and display the "About" page
    st.empty()
    about_page()

else:
    col4, col5 = st.columns([4,1])
    with col4:

        # Create the Streamlit app
        st.title('Multi-Modal ISL Recognition System')
        st.write('*A project for recognizing Indian Sign Language using computer vision and deep learning*')
        st.metric(label="Temperature in Guwahati, Assam, IN", value="27°C")

        # Create a date input component with a default value of today's date
        selected_date = date.today()

        # Display the selected date
        st.write('Today is:', selected_date)

        # st.subheader('Welcome!')



        # Dropdown menu box
        # st.subheader("Please select Detection Type")
        # main_options = ["Select an option", "Static Gesture", "Alphabets", "Digits"]
        # main_choice = st.sidebar.selectbox(" ", main_options)

        main_choice = st.sidebar.selectbox("Select Detection Type", ["Please Choose Here","Static Gesture","Alphabets", "Digits","Dynamic Gesture"])

        # Display quote texts in italics
        if main_choice == "Please Choose Here":
            st.markdown("<h3 style='{}'><strong>WELCOME!!</strong></h3>".format(centered_style), unsafe_allow_html=True)
            st.markdown("_\"The hands have a language of their own, expressing stories through graceful gestures\"_")
            st.markdown("_\"In the realm of silence, the hands speak the language of the heart\"_")
            st.markdown("_\"Every gesture holds a tale, a rich narrative woven by the hands of expression\"_")
            st.write(" ")
            st.write(" ")
            st.write('*Initializing the system...*')
            progress_bar = st.progress(0)
            for i in range(1, 11):
                time.sleep(0.2)
                progress_bar.progress(i / 10)
            st.write('*System Successfully Loaded*')


        else:
            # Clear everything on the screen except the title
            st.empty()

        #MAIN CHOICE STATIC GESTURE
        if main_choice == "Static Gesture":

            model_SG = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
            st.subheader("*Selected detection type is Static Gesture*")
            # sub_options = ["Select an option", "From Image", "From Camera", "From Webcam"]
            # sub_choice = st.sidebar.selectbox(" ", sub_options)

            sub_choice = st.sidebar.selectbox("Select Detection Source", ["Please Choose Here","From Image", "From Camera", "From Webcam"])

            # For Sub-option 1 (From Image)
            if sub_choice == "From Image":
                st.subheader('*Selected detection source is images*')
                uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
                # Process the uploaded image and run inference
                if uploaded_file is not None:
                    # Load the image
                    image = Image.open(uploaded_file)
                    
                    # Convert the image to OpenCV format
                    img_cv = np.array(image)
                    
                    # Run the YOLOv5 model on the image
                    with st.spinner('Running object detection...'):
                        results = model_SG(img_cv)
                    
                    # Display the image with the detected objects and their bounding boxes
                    st.image(results.render(), use_column_width=True)

                    # Display the class names of the detected objects
                    for obj in results.xyxy[0]:
                        label = f'{model_SG.names[int(obj[5])]} ({obj[4]:.2f})'
                        # st.write('Gesture shown in the above picture is: ', label)
                        st.write('Gesture shown in the above picture is: ', f'<span style="font-size:40px">{label}</span>', unsafe_allow_html=True)

            # For Sub-option 2 (From Camera)
            if sub_choice == "From Camera":
                # st.subheader("Selected Sub-option is From Camera")
                st.subheader("*Selected detection source is picture clicked from PC's WebCam*")
                if st.checkbox("Turn on WebCam"):
                    picture = st.camera_input("Take a picture")

                    if picture is not None:
                        try:            
                        # Load the image
                            image = Image.open(picture)
                            img_cv = np.array(image)
                            # Run the YOLOv5 model on the image
                            with st.spinner('Running object detection...'):
                                results = model_SG(img_cv)
                            # Display the image with the detected objects and their bounding boxes
                            st.image(results.render(), use_column_width=True)
                            # Display the class names of the detected objects
                            for obj in results.xyxy[0]:
                                label = f'{model_SG.names[int(obj[5])]} ({obj[4]:.2f})'
                                # st.write('Gesture shown in the above picture is: ', label)
                                st.write('Gesture shown in the above picture is: ', f'<span style="font-size:40px">{label}</span>', unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                    else:
                        st.warning("No image captured from the camera.")


            # For Sub-option 3 (From WebCam)
            if sub_choice == "From Webcam":
                st.subheader("*Selected detection source is PC's live WebCam feed*")
                run = st.checkbox('Turn On WebCam')
                video_feed = st.empty()

                # Define a function to capture frames from the webcam and run inference
                def run_object_detection():
                    cap = cv2.VideoCapture(0)
                    cap.set(3, 800)
                    cap.set(4, 600)
                    while run:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        with st.spinner('Running object detection...'):
                            results = model_SG(frame)

                        img_with_boxes = results.render()
                        video_feed.image(img_with_boxes, channels='BGR')

                    cap.release()

                if run:
                    run_object_detection()


        # For Option 2 (Alphabets)
        if main_choice == "Alphabets":
            model_alphabets = torch.hub.load('ultralytics/yolov5', 'custom', path='best_alphabet_main.pt', force_reload=True)
            st.subheader("*Selected Detection Type is Alphabets*")
            # sub_options = ["Select an option", "From Image", "From Camera", "From Webcam"]
            # sub_choice = st.sidebar.selectbox(" ", sub_options)

            sub_choice = st.sidebar.selectbox("Select Detection Source", ["Please Choose Here","From Image", "From Camera", "From Webcam"])

            # For Sub-option 1 (From Image)
            if sub_choice == "From Image":
                st.subheader('*Selected detection source is images*')
                uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
                # Process the uploaded image and run inference
                if uploaded_file is not None:
                    # Load the image
                    image = Image.open(uploaded_file)
                    
                    # Convert the image to OpenCV format
                    img_cv = np.array(image)
                    
                    # Run the YOLOv5 model on the image
                    with st.spinner('Running object detection...'):
                        results = model_alphabets(img_cv)
                    
                    # Display the image with the detected objects and their bounding boxes
                    st.image(results.render(), use_column_width=True)

                    # Display the class names of the detected objects
                    for obj in results.xyxy[0]:
                        label = f'{model_alphabets.names[int(obj[5])]} ({obj[4]:.2f})'
                        # st.write('Gesture shown in the above picture is: ', label)
                        st.write('Alphabet shown in the above picture is: ', f'<span style="font-size:40px">{label}</span>', unsafe_allow_html=True)

            # For Sub-option 2 (From Camera)
            if sub_choice == "From Camera":
                # st.subheader("Selected Sub-option is From Camera")
                st.subheader("*Selected detection source is picture clicked from PC's WebCam*")
                if st.checkbox("Turn on WebCam"):
                    picture = st.camera_input("Take a picture")
                    if picture is not None:
                        try:  
                            # Load the image
                            image = Image.open(picture)
                            img_cv = np.array(image)
                            # Run the YOLOv5 model on the image
                            with st.spinner('Running object detection...'):
                                results = model_alphabets(img_cv)
                            # Display the image with the detected objects and their bounding boxes
                            st.image(results.render(), use_column_width=True)
                            # Display the class names of the detected objects
                            for obj in results.xyxy[0]:
                                label = f'{model_alphabets.names[int(obj[5])]} ({obj[4]:.2f})'
                                # st.write('Gesture shown in the above picture is: ', label)
                                st.write('Alphabet shown in the above picture is: ', f'<span style="font-size:40px">{label}</span>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                    else:
                        st.warning("No image captured from the camera.")

            # For Sub-option 3 (From WebCam)
            if sub_choice == "From Webcam":
                st.subheader("*Selected detection source is PC's live WebCam feed*")
                run = st.checkbox('Turn On WebCam')
                video_feed = st.empty()

                # Define a function to capture frames from the webcam and run inference
                def run_object_detection():
                    cap = cv2.VideoCapture(0)
                    cap.set(3, 800)
                    cap.set(4, 600)
                    while run:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        with st.spinner('Running object detection...'):
                            results = model_alphabets(frame)

                        img_with_boxes = results.render()
                        video_feed.image(img_with_boxes, channels='BGR')

                    cap.release()

                if run:
                    run_object_detection()


        # For Option 3 (Digits)
        if main_choice == "Digits":
            model_digits = torch.hub.load('ultralytics/yolov5', 'custom', path='best_digit_final.pt', force_reload=True)
            st.subheader("*Selected Detection Type is Digits*")
            # sub_options = ["Select an option", "From Image", "From Camera", "From Webcam"]
            # sub_choice = st.sidebar.selectbox(" ", sub_options)

            sub_choice = st.sidebar.selectbox("Select Detection Source", ["Please Choose Here","From Image", "From Camera", "From Webcam"])

            # For Sub-option 1 (From Image)
            if sub_choice == "From Image":
                st.subheader('*Selected detection source is images*')
                uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
                # Process the uploaded image and run inference
                if uploaded_file is not None:
                    # Load the image
                    image = Image.open(uploaded_file)
                    
                    # Convert the image to OpenCV format
                    img_cv = np.array(image)
                    
                    # Run the YOLOv5 model on the image
                    with st.spinner('Running object detection...'):
                        results = model_digits(img_cv)
                    
                    # Display the image with the detected objects and their bounding boxes
                    st.image(results.render(), use_column_width=True)

                    # Display the class names of the detected objects
                    for obj in results.xyxy[0]:
                        label = f'{model_digits.names[int(obj[5])]} ({obj[4]:.2f})'
                        # st.write('Gesture shown in the above picture is: ', label)
                        st.write('Digit shown in the above picture is: ', f'<span style="font-size:40px">{label}</span>', unsafe_allow_html=True)

            # For Sub-option 2 (From Camera)
            if sub_choice == "From Camera":
                # st.subheader("Selected Sub-option is From Camera")
                st.subheader("*Selected detection source is picture clicked from PC's WebCam*")
                if st.checkbox("Turn on WebCam"):
                    picture = st.camera_input("Take a picture")
                    if picture is not None:
                        try:
                    # Load the image
                            image = Image.open(picture)
                            img_cv = np.array(image)
                            # Run the YOLOv5 model on the image
                            with st.spinner('Running object detection...'):
                                results = model_digits(img_cv)
                            # Display the image with the detected objects and their bounding boxes
                            st.image(results.render(), use_column_width=True)
                            # Display the class names of the detected objects
                            for obj in results.xyxy[0]:
                                label = f'{model_digits.names[int(obj[5])]} ({obj[4]:.2f})'
                                # st.write('Gesture shown in the above picture is: ', label)
                                st.write('Digit shown in the above picture is: ', f'<span style="font-size:40px">{label}</span>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                    else:
                        st.warning("No image captured from the camera.")


            # For Sub-option 3 (From WebCam)
            if sub_choice == "From Webcam":
                st.subheader("*Selected detection source is PC's live WebCam feed*")
                run = st.checkbox('Turn On WebCam')
                video_feed = st.empty()

                # Define a function to capture frames from the webcam and run inference
                def run_object_detection():
                    cap = cv2.VideoCapture(0)
                    cap.set(3, 800)
                    cap.set(4, 600)
                    while run:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        with st.spinner('Running object detection...'):
                            results = model_digits(frame)

                        img_with_boxes = results.render()
                        video_feed.image(img_with_boxes, channels='BGR')

                    cap.release()

                if run:
                    run_object_detection()

        #For Option 4 (Dynamic Gesture)
        if main_choice == "Dynamic Gesture":
            model_dynamic = torch.hub.load('ultralytics/yolov5', 'custom', path='best_dynamic.pt', force_reload=True)
            st.subheader("*Selected Detection Type is Dynamic Gestures*")
            sub_choice = st.sidebar.selectbox("Select Detection Source", ["Please Choose Here","From Video File", "From Webcam"])
            
            #For Video File Option
            if sub_choice == "From Video File":

                # Set confidence threshold for object detection
                conf_threshold = 0.5
                def detect_objects(model_dynamic, image):
                    # Convert image to numpy array
                    image_np = np.array(image)

                    # Perform object detection
                    results = model_dynamic(image_np)
                    results = results.pandas().xyxy[0]  # Access the predictions

                    return results

                uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov"])

                if uploaded_file is not None:
                    # Save the video file to a temporary location
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(uploaded_file.read())

                    # Load the video
                    video = cv2.VideoCapture(temp_file.name)

                    # Get video details
                    fps = video.get(cv2.CAP_PROP_FPS)
                    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    # Create an output video writer
                    output_path = "output.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

                    # Process each frame of the video
                    while True:
                        ret, frame = video.read()
                        if not ret:
                            break

                        # Convert frame to PIL Image
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                        # Detect objects in the frame
                        result = detect_objects(model_dynamic, image)

                        # Draw bounding boxes on the frame
                        for _, row in result.iterrows():
                            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                            label = row['name']
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), thickness=2)

                        # Write the frame to the output video
                        output_video.write(frame)

                    # Release the video writer and video capture objects
                    output_video.release()
                    video.release()

                    # Open the output video file
                    video_file = open(output_path, 'rb')
                    video_bytes = video_file.read()

                    # Display the output video in Streamlit
                    st.video(video_bytes)

                    # Remove the temporary video file
                    temp_file.close()
            if sub_choice == "From Webcam":
                st.subheader("*Selected detection source is PC's live WebCam feed*")
                run = st.checkbox('Turn On WebCam')
                video_feed = st.empty()

                # Define a function to capture frames from the webcam and run inference
                def run_object_detection():
                    cap = cv2.VideoCapture(0)
                    cap.set(3, 800)
                    cap.set(4, 600)
                    while run:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        with st.spinner('Running object detection...'):
                            results = model_dynamic(frame)

                        img_with_boxes = results.render()
                        video_feed.image(img_with_boxes, channels='BGR')  

                    cap.release()
                if run:
                    run_object_detection()




    with col5:
        st.write(" ")
        st.write(" ")
        st.write(" ")

        st.image("logo1.png", width=110)

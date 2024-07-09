import cv2
import numpy as np
import tensorflow as tf

# Load the Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(r'models/haarcascade_frontalface_default.xml')

# Load the pre-trained emotion classification model
# Download the pre-trained model from here
# https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
classifier = tf.keras.models.load_model(r'models/ResNet50_Transfer_Learning.keras')

# The list of emotion labels that our model was trained on. (Resnet50)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Start capturing video from the webcam (device 0 by default)
# If you want to apply model on any video define video path
video_path = "video_file_path"
cap = cv2.VideoCapture(0)

# Define the desired window size
desired_width = 1200
desired_height = 800

while True:
    # Read each frame from the video capture
    _, frame = cap.read()

    # Convert the frame to grayscale for the face detection because we trained our model on grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray)

    # Process each face detected (Multi face detection)
    for (x, y, w, h) in faces:
        # Draw a rectangle around each detected face    
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Extract the region of interest (ROI) as the face area from the grayscale frame
        roi_gray = gray[y:y+h, x:x+w]
        # Resize the ROI to the size expected by the model (224x224 pixels in this case)
        roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)

        # Proceed if the ROI is not empty
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0  # Normalize pixel values
            roi = np.stack((roi,)*3, axis=-1)  # Convert single channel to 3 channels
            roi = np.expand_dims(roi, axis=0)  # Add batch dimension

            # Predict the emotion of the face using the pre-trained model
            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)

            # Display the predicted emotion label on the frame
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Display message if no faces are detected
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Resize the frame to the desired window size
    frame = cv2.resize(frame, (desired_width, desired_height))

    # Show the frame with the detected faces and emotion labels
    cv2.imshow('Emotion Detector', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
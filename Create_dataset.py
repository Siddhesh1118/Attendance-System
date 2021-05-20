import cv2
import os

#For checking the existence of path
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


face_id = input('Enter Your Roll No:')
# Start capturing video for creating dataset
vid_cam = cv2.VideoCapture(0)

# Detect face/object in video stream using Haarcascade Frontal Face
face_detector = cv2.CascadeClassifier('C:/Users/User/Desktop/EXPERIMENTS/Automatic_attendence_system_using_facial_recognition_python_openCV-main/haarcascades/haarcascade_frontalface_default.xml')

# Prepare sample face image
count = 0
# Path to save the samples
assure_path_exists("C:/Users/User/Desktop/dataset")

# Start looping
while (True):

    # Capture video frame
    _, image_frame = vid_cam.read()

    # Convert samples to grayscale
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loops for each faces
    for (x, y, w, h) in faces:
        # Crop the image frame into rectangle
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Counting sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If collected sample count reaches 50, stop taking video or samples
    elif count >= 50:
        print("Successfully Captured")
        break

# Stop taking samples
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()

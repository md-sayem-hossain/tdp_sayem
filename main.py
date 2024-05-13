from scipy.spatial import distance  # Importing a specific function 'distance' from the module 'scipy.spatial'
from imutils import face_utils  # Importing the 'face_utils' module from 'imutils'
import imutils  # Importing the 'imutils' library for convenient image processing
import pygame  # Importing the 'pygame' library for multimedia applications
import dlib  # Importing the 'dlib' library for machine learning and computer vision
import cv2  # Importing the 'cv2' module for computer vision tasks
import threading  # Importing the 'threading' module for thread-based parallelism
import queue  # Importing the 'queue' module for implementing queues
from gtts import gTTS  # Importing the 'gTTS' module for text-to-speech synthesis
import numpy as np  # Importing the 'numpy' library for numerical computations
import sounddevice as sd  # Importing 'sounddevice' for recording and playing audio
import soundfile as sf  # Importing 'soundfile' for reading and writing sound files
from service.openai_service import ask_openai_question, speech_to_text  # Importing custom functions from a service module

two_request_flag = False  # Flag indicating if two requests have been made
request_in_progress = False  # Flag indicating if a request is currently in progress
alarm_on = False  # Flag indicating if the alarm is currently active
incorrect_answers = 0  # Counter for the number of incorrect answers given

def eye_aspect_ratio(eye):  # Function to calculate the eye aspect ratio
    A = distance.euclidean(eye[1], eye[5])  # Euclidean distance between points 1 and 5 of the eye
    B = distance.euclidean(eye[2], eye[4])  # Euclidean distance between points 2 and 4 of the eye
    C = distance.euclidean(eye[0], eye[3])  # Euclidean distance between points 0 and 3 of the eye
    ear = (A + B) / (2.0 * C)  # Calculating the eye aspect ratio using the distances
    return ear

def record_audio(duration=3):  # Function to record audio
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=2)  # Recording audio for the specified duration
    sd.wait()  # Wait for the recording to complete
    return np.mean(recording, axis=1)  # Returning the mean of the recorded audio data

def play_audio(filename):  # Function to play audio from a file
    data, fs = sf.read(filename, dtype='float32')  # Reading audio data from the file
    sd.play(data, fs)  # Playing the audio
    sd.wait()  # Waiting for the playback to complete

# Initializing Pygame for audio playback
pygame.init()
pygame.mixer.init()

def alarm_volume_increase():  # Function to increase alarm volume gradually
    volume = 0.1  # Starting volume (0 to 1)
    pygame.mixer.music.set_volume(volume)  # Setting the initial volume
    while volume < 1.0 and pygame.mixer.music.get_busy():  # Gradually increasing volume till maximum or until sound finishes
        volume += 0.1  # Incrementing volume
        pygame.mixer.music.set_volume(volume)  # Setting the new volume
        pygame.time.delay(1000)  # Delaying for one second before increasing volume

def start_alarm():  # Function to start the alarm
    pygame.mixer.music.load("sounds/emergency.mp3")  # Loading the alarm sound
    pygame.mixer.music.play(-1)  # Playing the alarm sound in a loop
    threading.Thread(target=alarm_volume_increase, daemon=True).start()  # Starting a thread to increase the alarm volume gradually

def stop_alarm():  # Function to stop the alarm
    pygame.mixer.music.stop()  # Stopping the alarm sound

def process_drowsiness(ear):  # Function to process drowsiness detection
    global incorrect_answers  # Accessing the global variable
    global two_request_flag  # Accessing the global variable
    global request_in_progress  # Accessing the global variable
    global alarm_on  # Accessing the global variable

    request_in_progress = True  # Setting the request in progress flag to True

    question = ask_openai_question()  # Asynchronous API call to ask a question
    print("Question: ", question)  # Printing the question
    tts = gTTS(text=question, lang='en')  # Generating text-to-speech from the question
    tts.save("question.mp3")  # Saving the generated speech to a file
    play_audio("question.mp3")  # Playing the question audio

    print("....recording start....")  # Printing status message
    recording = record_audio(4)  # Recording audio for 4 seconds
    sf.write('recording.wav', recording, 44100)  # Writing the recorded audio to a file
    print("....recording end....")  # Printing status message

    if speech_to_text('recording.wav'):  # Checking if no reply is received
        incorrect_answers += 1  # Incrementing incorrect answers count
    else:
        incorrect_answers = 0  # Resetting incorrect answers count
    print("incorrect_answers: ", incorrect_answers)  # Printing the number of incorrect answers

    if incorrect_answers >= 2:  # Checking if two incorrect answers are received
        print("two incorrect answers.....................")  # Printing status message
        two_request_flag = True  # Setting the two request flag to True
        alarm_on = True  # Setting the alarm flag to True
        start_alarm()  # Starting the alarm
    request_in_progress = False  # Resetting the request in progress flag

def handle_drowsiness():  # Function to handle drowsiness detection
    global incorrect_answers  # Accessing the global variable

    while True:  # Running indefinitely
        ear = ear_queue.get()  # Getting eye aspect ratio from the queue

        if ear < thresh and not request_in_progress and not two_request_flag:  # Checking conditions for drowsiness
            threading.Thread(target=process_drowsiness, args=(ear,)).start()  # Starting a thread to process drowsiness

thresh = 0.25  # Threshold for detecting drowsiness
frame_check = 20  # Number of consecutive frames for which the condition must hold for triggering drowsiness
detect = dlib.get_frontal_face_detector()  # Initializing face detector
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Initializing landmark predictor
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]  # Defining indices for left eye
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]  # Defining indices for right eye
cap = cv2.VideoCapture(0)  # Initializing video capture from default camera
ear_queue = queue.Queue()  # Initializing a queue for storing eye aspect ratios
flag = 0  # Initializing a counter variable
flag2 = 0  # Initializing another counter variable

# Starting the drowsiness handler thread
threading.Thread(target=handle_drowsiness).start()

while True:  # Running indefinitely
    ret, frame = cap.read()  # Reading a frame from the camera
    frame = imutils.resize(frame, width=450)  # Resizing the frame for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting the frame to grayscale
    subjects = detect(gray, 0)  # Detecting faces in the grayscale frame

    for subject in subjects:  # Iterating over detected faces
        shape = predict(gray, subject)  # Predicting facial landmarks
        shape = face_utils.shape_to_np(shape)  # Converting landmarks to NumPy array
        leftEye = shape[lStart:lEnd]  # Extracting left eye landmarks
        rightEye = shape[rStart:rEnd]  # Extracting right eye landmarks
        leftEAR = eye_aspect_ratio(leftEye)  # Calculating left eye aspect ratio
        rightEAR = eye_aspect_ratio(rightEye)  # Calculating right eye aspect ratio
        ear = (leftEAR + rightEAR) / 2.0  # Calculating average eye aspect ratio
        leftEyeHull = cv2.convexHull(leftEye)  # Creating convex hull around left eye
        rightEyeHull = cv2.convexHull(rightEye)  # Creating convex hull around right eye
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)  # Drawing contours around left eye
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)  # Drawing contours around right eye

        if ear < thresh:  # Checking if eye aspect ratio is below threshold
            flag2 = 0  # Resetting the counter variable
            flag += 1  # Incrementing the counter variable
            if flag >= frame_check and not request_in_progress and not two_request_flag:  # Checking if drowsiness conditions are met
                ear_queue.put(ear)  # Putting eye aspect ratio in the queue
        else:  # If eye aspect ratio is above threshold
            flag = 0  # Resetting the counter variable
            flag2 += 1  # Incrementing the counter variable
            incorrect_answers = 0  # Resetting the incorrect answers count
            if alarm_on and flag2 >= frame_check:  # Checking if alarm is on and conditions are met to stop it
                alarm_on = False  # Resetting the alarm flag
                two_request_flag = False  # Resetting the two request flag
                stop_alarm()  # Stopping the alarm

    cv2.imshow("Frame", frame)  # Displaying the frame with drawn contours
    key = cv2.waitKey(1) & 0xFF  # Waiting for key press
    if key == ord("q"):  # If 'q' is pressed, exit the loop
        break

cv2.destroyAllWindows()  # Closing all OpenCV windows
cap.release()  # Releasing the video capture object

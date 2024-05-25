import cv2
from playsound import playsound

# Opening the camera
cap = cv2.VideoCapture(0)

# import the face cascade the pre trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize object tracker
tracker = cv2.TrackerCSRT_create()

# check if the tracker is initialized or not
tracker_initialized = False
flag = 0        # made for checking if the alarm is triggered or not ( 1 means trigerred and 0 means not triggered)

# Set the position of the vertical line (center of the frame)
ret, frame1 = cap.read()
height1, width1, _ = frame1.shape
# Draw line in the frame
cv2.line(frame1, (width1//2, 0), (width1//2, height1), (0, 255, 0), 1)      ## takes frame , start point (x,y) and end point (x,y) and the color and line width as pixel

line_position = width1 // 2

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # for the dimensions of the frame
    height, width, _ = frame.shape


    # Detect faces if tracker is not initialized             ### necessary if the tracker is already initilized for a particular run then it avoids the unnecessary initialisation every time.
    if not tracker_initialized:
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))     # could not understand the meanign so just copied the line
        for (x, y, w, h) in faces:
            tracker.init(frame, (x, y, w, h))
            tracker_initialized = True

    # Update tracker for each face
    if tracker_initialized:
        success, box = tracker.update(frame)        ## stores the location and the dimension of the tracker in a box variable
        if success:
            (x, y, w, h) = [int(v) for v in box]        ## breaks the touple of x,y,w,h and make them seprate variables as to form a rectangle we want integer values not the float
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the center of the bounding box
            
            center_x = x + w // 2

            # Check if the center of the bounding box crosses the line
            if center_x > line_position:
                # activates the alarm when crossed
                cv2.putText(frame, 'Alarm: Crossed the line!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                flag = 1
                

    cv2.imshow('Face Movement Tracker', frame)
    if flag ==1 :
        playsound('alarm.wav')
    # Check for the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break


# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

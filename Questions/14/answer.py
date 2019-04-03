import cv2

# Initialize the camera
cap = cv2.VideoCapture(0)

while 1:
    # Capture each frame
    ret, frame = cap.read()

    # Convert the frame from bgr to gray
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Show the result
    cv2.imshow('Video', grayscale_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

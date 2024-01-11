import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the pre-trained facial features detection model
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

# Load the image
image = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Iterate over each detected face
for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Extract the region of interest (ROI) within the face
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    
    # Detect eyes in the ROI
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    # Iterate over each detected eye
    for (ex, ey, ew, eh) in eyes:
        # Draw a rectangle around the eye
        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
    
    # Detect nose in the ROI
    nose = nose_cascade.detectMultiScale(roi_gray)
    
    # Iterate over each detected nose
    for (nx, ny, nw, nh) in nose:
        # Draw a rectangle around the nose
        cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)
    
    # Detect mouth in the ROI
    mouth = mouth_cascade.detectMultiScale(roi_gray)
    
    # Iterate over each detected mouth
    for (mx, my, mw, mh) in mouth:
        # Draw a rectangle around the mouth
        cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 255), 2)

# Display the image with the detected faces and facial features
cv2.imshow('Face and Facial Features Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

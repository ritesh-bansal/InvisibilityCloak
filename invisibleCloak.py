import numpy as np
import cv2


def detectBlue(frame, background):
    # Convert image to HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Create HSV color mask and segment the image based on Blue color
    sensitivity = 60
    HValue = 20  # Change this value if you want to segment some other color
    lightBlue = np.array([HValue - sensitivity, 60, 60])
    darkBlue = np.array([HValue + sensitivity, 255, 255])
    mask = cv2.inRange(hsvImage, lightBlue, darkBlue)  # Creating a segmentation mask for the blue color

    # Apply closing operation to fill out the unwanted gaps in the image. Bigger the kernel size, lesser the gaps
    kernelSize = 10
    kernel = np.ones((kernelSize, kernelSize), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find the contour coordinates of the biggest area and create a mask of that area
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contSorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    contourMask = cv2.fillPoly(np.zeros((500, 500, 3), dtype=np.uint8), pts=[contSorted[0]], color=(255, 255, 255))

    # create the two masks with the background image and the main object mask such that we can superimpose them together
    objectMask = cv2.fillPoly(frame, pts=[contSorted[0]], color=(0, 0, 0))
    backgroundMask = np.bitwise_and(contourMask, background)

    # Final image is created by doing a bitwise and of the two masks, which in turn removes the color in question
    # and replaces it with the background
    finalImg = cv2.bitwise_or(objectMask, backgroundMask)

    return finalImg


# Initiate video capture from source '0.
# Change the source value if you have more than one webcam and wish to use the secondary one
cap = cv2.VideoCapture(0)

# Read the background initially, resize it and then show it, wait for the user to press a key before continuing
ret, background = cap.read()
background = cv2.resize(background, (500, 500))
cv2.imshow('Background', background)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the video in MP4 format in the same directory as the code (Optional)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (500, 500))

while True:
    # Capture frame-by-frame, resize and then apply the invisibility algorithm
    ret, frame = cap.read()
    frame = cv2.resize(frame, (500, 500))
    image = detectBlue(frame, background)

    # The processed frame is added to the video
    out.write(image)

    # Display the resulting video
    cv2.imshow('Image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy all assets
cap.release()
out.release()
cv2.destroyAllWindows()
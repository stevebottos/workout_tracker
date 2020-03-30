import cv2

path_to_videos = "./data/"

filename = path_to_videos + "IMG_0500.mp4"

cap = cv2.VideoCapture(filename)

# Check if camera opened successfully
if (not cap.isOpened()):
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # flip 180 degrees
        frame = cv2.flip(frame, flipCode=-1)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

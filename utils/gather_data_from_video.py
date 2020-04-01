import cv2
import os

output_folder = "../data/unsorted_images/"

# Video capture stuff
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

with open("start_from.txt", "r") as f:
    img_count = int(f.readline())

while True:
    _, frame = cap.read()
    cv2.imwrite(output_folder + str(img_count) + ".png", frame)
    img_count += 1
    key = cv2.waitKey(1)

    cv2.imshow('', frame)

    with open("start_from.txt", "w") as f:
        f.write(str(img_count))

    if key == 32:  # 32 is space key on my computer.
        break

cap.release()
cv2.destroyAllWindows()

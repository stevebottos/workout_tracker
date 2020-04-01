import cv2
import os

input_folder= "../data/unsorted_images/"

out_class1 = "../data/train/class1/"  # This class represents the "top" position of a pushup
out_class2 = "../data/train/class2/"  # This class represents the "bottom" position of a pushup
out_class3 = "../data/train/class3/"  # This class represents anything that doesn't fit in the other two categories

for f in os.listdir(input_folder):
    im = cv2.imread(input_folder + f)

    cv2.imshow("", im)
    opt = cv2.waitKey()

    if opt == 49:
        cv2.imwrite(out_class1 + str(f) + "_labeled.png", im)
    elif opt == 50:
        cv2.imwrite(out_class2 + str(f) + "_labeled.png", im)
    elif opt == 51:
        cv2.imwrite(out_class3 + str(f) + "_labeled.png", im)

    os.remove(input_folder + f)
    cv2.destroyAllWindows()

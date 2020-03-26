import cv2
import os

input_folder = "unsorted_images"
out_class1 = "train/class1"  # This class represents the "top" position of a pushup
out_class2 = "train/class2"  # This class represents the "bottom" position of a pushup
out_class2 = "train/class3"  # This class represents anything that doesn't fit in the other two categories

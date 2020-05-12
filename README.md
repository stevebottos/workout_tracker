# workout_tracker

Need to make these folders locally to store training images:
unsorted_images  
train/class1  
train/class2  
train/class3

And for testing:
test/raw_images

**gather_data_from_video.py** starts a connection to your webcam and saves each frame to unsorted_images (it uses **start_from.txt** to ensure that no two images will have the same filename if you start and stop recording multiple times), from there use **sort_data.py** to annotate. Then, **model.py** has everything you need to get started with training and testing

For everyone asking for the model: https://drive.google.com/file/d/15HYJKqIi43ZfQjrvNp-5e7r0EqQoTkHT/view?usp=sharing

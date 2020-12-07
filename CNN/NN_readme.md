# All Neural network files and supporting notebooks are stored within this directory

## Data Collection

- The emulator was run with the screen recorded, completing a single lap in each clip
- each clip satisfies conditions required by it's relevant class
- The .mp4 files used were recorded in zoom and trimmed appropriately in order to contain only the desired class frames. 
- The title of each mp4 file is the class that it represents

## Data Processing

 - The jupyter notebook `video_to_image_data.ipynb` uses OpenCV to parse the mp4 files into individual frames. 
 - Frames are partitioned into the `dataset/<class>/` directory, and are saved as `frame<number>.jpg` within the corresponding directory
 - Due to the file sizes of the dataset, instead, a public link to the Google Drive is provided. Within the drive folder is the raw .mp4 files uses, as well as a .7z compression of the dataset. Link: https://drive.google.com/drive/folders/16hwd7d2v04Ze3ib3D7B_ipzY2KHTiEy7?usp=sharing. 
 - In total, slighly less than 60,000 image frames are produced.
 
## Data Modeling

 - The jupyter notebook `construct_frame_classifier.ipynb` uses Tensorflow Keras to train a convolutional neural network on the image frames processed above
 - The notebook further preprocesses each image frame by converting the images to grayscale, and resizing the images to 80px wide by 64px tall
 - The CNN is then trained using a batch-size of 128 for 100 epochs 
 - We have trained the model 3 separate times, and stored each as a separate .h5 file, which can be loaded from later
 - `model.h5` is the simplest of the 3, considering `center'`, `near_left`, `near_right`, `off_left`, and `off_right` classes. This model achieved a validation accuracy of 0.9966
 - `model_v2.h5` is a sligtly more complex than the previous. including the same classes in addition to `wall_left` and `wall_right`. This model achieved a validation accuracy of 0.9974
 - `model_v3.h5` being the most complex of the 3, it includes each class that has a video file present for. The model achieved a validation accuracy of 0.9980
 - `classifier_v4.h5` being a further updated version of the classifiers, which has been trained on a more complete dataset which utilizes both gamecube and n64 video frames for training and validation. 
 
 

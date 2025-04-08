YOLO (You Only Look Once) is a very powerful and a fast algorithm in object detection. A strong understanding of the algorithm is essential before we start to code.

It contains three files:
    * coco.names
    * yolov3.cfg
    * yolov3.weights

Make sure you have installed 
pip install numpy
pip install opencv-python

Create a folder images and have some pictures inside it to test the object detection.
The main.py has the script to detect the objects in the images which are inside the "images" folder.
You can now run the file by giving this command on your command promt
python main.py --image images/apples-and-oranges.png


The main1.py has the script to detect the objects in realtime via webcam.
Program can be run by giving the command python run main.py .

Press q to quit the window of the image showing object detection

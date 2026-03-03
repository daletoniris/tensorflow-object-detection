
## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import time
import os
import cv2
import numpy as np
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import tensorflow as tf
import argparse
import sys
import re
# Set up camera constants
#IM_WIDTH = 1280
#IM_HEIGHT = 720
IM_WIDTH = 640    #Use smaller resolution for
IM_HEIGHT = 480   #slightly faster framerate

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'otro'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'otro','label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 81

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

### USB webcam ###
elif camera_type == 'usb':
    # Initialize USB webcam feed
    camera = cv2.VideoCapture("http://192.168.1.104:8080/?action=stream")
    #camera = cv2.VideoCapture(0)    
    ret = camera.set(3,IM_WIDTH)
    ret = camera.set(4,IM_HEIGHT)

    while(True):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = camera.read()
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
        
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        #human_string = label_lines[node_id]
        #if  np.squeeze(classes).astype(np.int32) == ("ochenta"):
         #  os.system("ssh pi@192.168.1.104 python /home/pi/robot3.py")
        
        #if "persona" in choclo:
        hash = []
        vocales = 'persona'
        persona = ["","","","",""]
        i = 0        
        for index, value in enumerate(classes[0]):
          object_dict = {}
          if scores[0, index] > 0.5:
            object_dict[(category_index.get(value)).get('name').encode('utf8')] = \
                        scores[0, index]
            hash.append(object_dict)
            #print (object_dict[(category_index.get(value)).get('name').encode('utf8')]) 
               #print ("hay una persona")       
        #print (str(objects))
        #objects1 = objects.astype(np.float) 
# printing final result 
        #print ("final array", str(objects)) 
            #lista_A1 = [nombre for nombre in objects if nombre.endswith('1.0')] 
            #if ({b'persona':1.0} in objects) == True:
            #
            #for letra in objects:
                #if re.match(letra, vocales):
            #print (hash)
            for elemento in hash:
               persona[i] = (str(elemento))    
               persona[i] = persona[i].split("'")
               i = i + 1 
            j = 0
            #while  j < len(persona):
    #pregunto si el string 'persona' esta dentro del elemento
            if(('persona' in persona[j][1]) == True):
                 print("ELEMENTO: ", j)
                 #os.system("ssh pi@192.168.1.104 python3 /home/pi/robotstop.py")
                 time.sleep(10)
                 print("PERSONA EXISTE EN persona[j][1]: "," j: ",j," ",persona[j][1])
            if(('ochenta' in persona[j][1]) == True):
                 #os.system("ssh pi@192.168.1.104 python3 /home/pi/robot4.py")
                 #time.sleep(10)                 
                 print("ELEMENTO: ", j)
                 print("OCHENTA EXISTE EN persona[j][1]: "," j: ",j," ",persona[j][1])
        #reemplazo los caracteres que no me sirven por espacios en blanco
                 #persona[j][2] = persona[j][2].replace(':', ' ')
                 #persona[j][2] = persona[j][2].replace('}', ' ')
        #recorro cada elemento para extraer el valor float
                 #for i in persona[j][2].split():
                  #try:
                #trata de convertir a flotante
                   #result = float(i)
                #rompe el ciclo ante el primer numero que logra convetir con exito
                   #break
                  #except:
                   #continue
                 # print("FLOAT OBTENIDO EN persona[j][2]: ",persona[j][2])
                 # print("VALOR LIMPIO: ",persona[j][2])
                 # print("FLOAT ENCONTRADO: ",result)
        #consulto si el valor esta dentro del rango aceptable
                  #if(result <= 1.0) & (result >= 0.8):
                  #   print("EL VALOR ESTA DENTRO DEL RANGO ACEPTABLE")
                  #else:
                   #print("EL VALOR NO ES ACEPTABLE")
           # j = j + 1
        #if ("{b'persona': 0.9979813}") in str(objects) == (True):
         #    print ("persona")      
        #print (category_index.get)
        # Press 'q' to quit
        #print (enumerate(classes[0]))
        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()

cv2.destroyAllWindows()


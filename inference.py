#####################################################################################
#python3 generate_tfrecord.py --image_dir=labelled_data  --xml_dir=labelled_data --labels_path=ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/label_map.pbtxt --output_path=ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/train.record --csv_path=ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/train.csv
#####################################################################################

from calculatemAP import MeanAveragePrecision
import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image
import tensorflow as tf
import cv2
import Inferencetf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2700)])

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import colab_utils
from object_detection.builders import model_builder
from object_detection.utils import object_detection_evaluation

import helper 
from helper import load_image_into_numpy_array
import TrainModel
import gc
from matplotlib import pyplot as plt

num_classes = 10

def load_image_into_numpy_array(path):
  """Load an image from file into a numpy array.

  Puts image into numpy array to feed into tensorflow graph.
  Note that by convention we put it into a numpy array with shape
  (height, width, channels), where channels=3 for RGB.

  Args:
    path: the file path to the image

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
  """
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def get_keypoint_tuples(eval_config):
  """Return a tuple list of keypoint edges from the eval config.
  
  Args:
    eval_config: an eval config containing the keypoint edges
  
  Returns:
    a list of edge tuples, each in the format (start, end)
  """
  tuple_list = []
  kp_list = eval_config.keypoint_edge
  for edge in kp_list:
    tuple_list.append((edge.start, edge.end))
  return tuple_list

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

def ReadFromVideo(detect_fn,category_index):
  cap = cv2.VideoCapture(2) 
  while(True):
    ret, frame = cap.read()
  
    if ret == True: 
      #frame = cv2.resize(frame, (640, 480)) 
      image_np = np.array(frame).reshape((640, 480, 3)).astype(np.uint8)
      input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
      detections, predictions_dict, shapes = detect_fn(input_tensor)

      image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np,
          detections['detection_boxes'][0].numpy(),
          (detections['detection_classes'][0].numpy() + 1).astype(int),  #Add one is mandatory
          detections['detection_scores'][0].numpy(),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=25,
          min_score_thresh=.10,
          agnostic_mode=False,
          skip_scores=True)
    
      image_np_with_detections = cv2.cvtColor(image_np_with_detections,cv2.COLOR_BGR2RGB)
      # Display the resulting frame    
      cv2.imshow('Started Recording...',image_np_with_detections)
  
      # Press Q on keyboard to stop recording
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
    # Break the loop
    else:
      break 
  
  # When everything done, release the video capture and video write objects
  cap.release()
  
  # Closes all the frames
  cv2.destroyAllWindows()

def GetActiveCameraStreamPorts():
    cams_test = 20
    active_devices = []
    for i in range(0, cams_test):
        cap = cv2.VideoCapture(i)
        test, frame = cap.read()
        if test:
            active_devices.append(int(i))
            #print("i : "+str(i)+" /// result: "+str(test))
    active_devices.sort()
        
    return active_devices


category_index = {
        1:{'id': 1,'name': 'candy_minipralines_lindt'},
        2:{'id': 2,'name': 'cereal_cheerios_honeynut'},
        3:{'id': 3,'name': 'cleaning_snuggle_henkel'},
        4:{'id': 4,'name': 'craft_yarn_caron'},
        5:{'id': 5,'name': 'drink_greentea_itoen'},
        6:{'id': 6,'name': 'drink_whippingcream_lucerne'},
        7:{'id': 7,'name': 'lotion_essentially_nivea'},
        8:{'id': 8,'name': 'pasta_lasagne_barilla'},
        9:{'id': 9,'name': 'snack_granolabar_naturevalley'},
        10:{'id': 10,'name': 'snack_biscotti_ghiott'},
      }

if __name__ == "__main__":

    #list_cams =  GetActiveCameraStreamPorts()
    
    # Load images and visualize
    #train_Image_Dir = '/home/gowtham/Desktop/localtests/Tensorflow2_experiments/Kerasbg/labelled_data'
    labels_file_path = '/home/gowtham/Desktop/Unity3d/Kerasbg/pretrained/label_map.pbtxt'
    pipeline_config = '/home/gowtham/Desktop/Unity3d/Kerasbg/pretrained/pipeline.config'
    checkpoint_path = '/home/gowtham/Desktop/Unity3d/Kerasbg/weights/ckpt29900-1'
    image_path = "/home/gowtham/Desktop/Unity3d/Kerasbg/labelled_data/rgb_2.png"
    test_folder_path = "/home/gowtham/Desktop/Unity3d/Kerasbg/labelled_data/Test"

    print('Sample Inference model.')
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(checkpoint_path).expect_partial()
    del(ckpt)

    '''
    label_map = label_map_util.load_labelmap(labels_file_path)
    categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=label_map_util.get_max_label_map_index(label_map),use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    del(label_map)
    del(categories)
    '''
    

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)
    gc.collect()
    mAP = MeanAveragePrecision(category_index)
    
    # Detections and groundtruth for image 1.


    #cap = cv2.VideoCapture("./videos/product_detection.avi")
    cap = cv2.VideoCapture(image_path) 
    i = 0
    #while(True):
    for image_path in glob.glob(test_folder_path + '/*.png'):
      cap = cv2.VideoCapture(image_path)
      fileName = os.path.split(image_path)[-1] 
      groundTruthValues = helper.xml_to_CustomTensorArray_file(image_path.replace("png","xml"))
      ret, frame = cap.read()
      i = i + 1
      if ret == True: 
        #frame = cv2.resize(frame, (640, 480)) 
        image_np = np.array(frame).reshape((frame.shape[0], frame.shape[1], 3)).astype(np.uint8)
        #image_np = np.fliplr(image_np)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        #detections, predictions_dict, shapes = detect_fn(input_tensor)
        detections,predictions_dict = Inferencetf.detect(input_tensor,detection_model)

        image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + 1).astype(int),  #Adding one is mandatory
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=25,
            min_score_thresh=.10)
      
        #image_np_with_detections = cv2.cvtColor(image_np_with_detections,cv2.COLOR_BGR2RGB)
        # Display the resulting frame    
        mAP(groundTruthValues,detections)
        
        #cv2.imshow('Started Recording...',image_np_with_detections)
        cv2.imwrite("./videos/"+fileName,image_np_with_detections)

        del(image_np)
        del(input_tensor)
        del(detections)
        del(predictions_dict)
        del(image_np_with_detections)
        gc.collect()
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
          
      # Break the loop
      else:
        break 
    
    mAP.CalculatemAP()
    # When everything done, release the video capture and video write objects
    cap.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
    




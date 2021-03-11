#####################################################################################
#python3 generate_tfrecord.py --image_dir=labelled_data  --xml_dir=labelled_data --labels_path=ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/label_map.pbtxt --output_path=ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/train.record --csv_path=ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/train.csv
#####################################################################################
'''

Train image is used as test image

1. Fetaure extraction is hard since train images are very less in size
		Try resize and crop method explained by professor and also change the bounding box offset.
2. Shadow are appearing done
3. rotation angle control
4. Calculate iou
'''

import numpy as np
import tensorflow as tf
import Inferencetf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.config.experimental.set_virtual_device_configuration(gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2700)])

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

import helper
import TrainModel
import gc


num_classes = 10
batch_size = 8
learning_rate = 0.01 
num_batches = 30000

def InferenceTest():
  print('Sample Inference model.')
  labels_file_path = '/home/gowtham/Desktop/Unity3d/Kerasbg/pretrained/label_map.pbtxt'
  model_dir = "/home/gowtham/Desktop/Unity3d/Kerasbg/weights/"
  pipeline_config = '/home/gowtham/Desktop/Unity3d/Kerasbg/pretrained/' + 'pipeline.config'
  
  checkpoint_path = model_dir + 'ckpt18800-1'
  configs = config_util.get_configs_from_pipeline_file(pipeline_config)
  model_config = configs['model']

  detection_model = model_builder.build(model_config=model_config, is_training=False)
  ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
  ckpt.restore(checkpoint_path)

  test_Images_NP = helper.load_image_into_numpy_array(train_Image_Dir+"/Test")

  label_id_offset = 1
  label_map = label_map_util.load_labelmap(labels_file_path)
  categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=label_map_util.get_max_label_map_index(label_map),use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  return test_Images_NP,category_index,detection_model


if __name__ == "__main__":
    # Load images and visualize
    train_Image_Dir = '/home/gowtham/Desktop/Unity3d/Kerasbg/labelled_data'
    #labels_file_path = '/home/gowtham/Desktop/localtests/Tensorflow2_experiments/Kerasbg/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/label_map.pbtxt'
    labels_file_path = '/home/gowtham/Desktop/Unity3d/Kerasbg/pretrained/labels.txt'
    checkpoint_path = "/home/gowtham/Desktop/Unity3d/Kerasbg/weights/"
    #train_csv_path= "/home/gowtham/Desktop/localtests/livellotrain/training_data_openvino/"
    model_dir = "/home/gowtham/Desktop/Unity3d/Kerasbg/pretrained/"
    train = True

    #helper.ReadCSVFile(train_csv_path+ "train.csv")
    

    if(train == True):
      category_index = helper.FileRead(labels_file_path)
      print('Finished importing Label file.')

      print('Started reading images.')
      train_Images_NP = helper.load_image_into_numpy_array(train_Image_Dir)
      all_keys = list(range(len(train_Images_NP)))
      print("Finished reading images.")

      print('Started reading xml annotations and images as numpy array.')
      getXMLAnnotationsAndImages = helper.xml_to_CustomTensorArray_folder(train_Image_Dir)
      helper.AnnotationsSummary(category_index,getXMLAnnotationsAndImages)
      print('Finished reading xml annotations.')

      print('Preparing Tensors...')
      TrainModel.PrepTensors(num_classes,getXMLAnnotationsAndImages,train_Images_NP,category_index)
      del(train_Images_NP)
      del(getXMLAnnotationsAndImages)
      gc.collect()
      print('Finished preparing Tensors')

      print("Started to create model")
      detection_model = TrainModel.CreateModel(model_dir)
      print("Finished creating model")

      print('Started training model.')
      TrainModel.TrainModelTF(batch_size,num_batches,learning_rate,detection_model,all_keys)
      print('Finished training model')
      
      print("Done.")

"""   
    test_Images_NP, category_index,detection_model = InferenceTest()
    for i in range(len(test_Images_NP)):
      test_Images_Tensor = tf.expand_dims(tf.convert_to_tensor(test_Images_NP[i][1], dtype=tf.float32), axis=0)
      input_tensor = tf.convert_to_tensor(test_Images_Tensor, dtype=tf.float32)
        
      detections,prediction_dict = Inferencetf.detect(input_tensor,detection_model)
      predclass = detections['detection_classes'][0].numpy().astype(np.uint32) + 1

      Inferencetf.plot_detections(
          test_Images_NP[i][1],
          detections['detection_boxes'][0].numpy(),
          predclass,
          detections['detection_scores'][0].numpy(),
          #category_index, figsize=(15, 20))
          category_index, figsize=(15,20), image_name="gif_frame_" + ('%02d' % i) + ".jpg") """

      #TrainModel.GetAccuracy(detection_model,test_Images_NP[i][1],batch_size,prediction_dict)
    





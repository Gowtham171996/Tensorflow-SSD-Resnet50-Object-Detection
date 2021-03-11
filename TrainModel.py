import tensorflow as tf
import numpy as np
import random
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as viz_utils
import datetime
import inference
import helper
import gc
import cv2

train_image_tensors_names = []
gt_classes_one_hot_tensors = []
gt_box_tensors = []
train_image_tensors_dict = {}

def PrepTensors(num_classes,getXMLAnnotationsAndImages,train_images_np,category_index):
    label_id_offset = 1
    with tf.device('/cpu:0'):
        for row in train_images_np:
            train_image_tensors_dict[row[0]] =  tf.expand_dims(tf.convert_to_tensor(row[1], dtype=tf.float32), axis=0)
    
    for row in getXMLAnnotationsAndImages:
        train_image_tensors_names.append(row.filename)
        gt_box_tensors.append(tf.convert_to_tensor(row.gtboxNP, dtype=tf.float32))
        classID = np.array([category_index[row.classname]],dtype= np.int32) - label_id_offset
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(classID)
        gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes))

    del(getXMLAnnotationsAndImages)
    del(train_images_np)
    gc.collect()
    print('Done prepping data.')

def CreateModel(model_dir):
    tf.keras.backend.clear_session()
    print('Building model and restoring weights for fine-tuning...', flush=True)
    
    pipeline_config = model_dir + 'pipeline.config'
    checkpoint_path = model_dir + 'checkpoint/ckpt-0'
    
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    #model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(model_config=model_config, is_training=True)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(_base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,_box_prediction_head=detection_model._box_predictor._box_prediction_head,)
    fake_model = tf.compat.v2.train.Checkpoint(_feature_extractor=detection_model._feature_extractor,_box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(checkpoint_path).expect_partial()
    

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')
    return detection_model


def TrainModelTF(batch_size, num_batches, learning_rate ,detection_model,train_images_np):
    tf.keras.backend.set_learning_phase(True)

    trainable_variables = detection_model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = [
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
        #,'WeightSharedConvolutionalBoxPredictor/ClassPredictionTower/conv2d_3/BatchNorm/feature_4']
    for var in trainable_variables:
        #print(var.name)
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    train_step_fn = get_model_train_step_function(detection_model, optimizer, to_fine_tune,batch_size,detection_model)
    print('Start fine-tuning!', flush=True)
    
    del(trainable_variables)
    del(to_fine_tune)
    gc.collect()
    start_time = datetime.datetime.now()
    class_loss = []
    local_loss = []
    for idx in range(num_batches):
        # Grab keys for a random subset of examples
        all_keys = list(range(len(train_images_np)))
        random.shuffle(all_keys)
        example_keys = all_keys[:batch_size]

        #total_loss = []
        gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
        gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
        image_tensors_name = [train_image_tensors_names[key] for key in example_keys]
        image_tensors = []
        [image_tensors.append(train_image_tensors_dict.get(image_tensors_name[i])) for i in range(len(example_keys))]
        
        # Training step (forward pass + backwards pass)
        total_loss_batch = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)
        local_loss.append(total_loss_batch[0].numpy())
        class_loss.append(total_loss_batch[1].numpy())
        
        if idx % 100 == 0:
            total_time = datetime.datetime.now() - start_time
            print('batch ' + str(idx) + ' of ' + str(num_batches) + ',localisation loss=' +  str(sum(local_loss)/len(local_loss))+',Classification loss=' +  str(sum(class_loss)/len(class_loss)) + ' Time=' + str(total_time), flush=True)
            checkpoint_path = "weights"
            ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
            manager = tf.train.CheckpointManager(ckpt,directory=checkpoint_path, max_to_keep=5,checkpoint_name='ckpt'+str(idx))
            manager.save()
            
            InferRandomImage(image_tensors,detection_model,total_loss_batch[2],batch_size,image_tensors_name[0],idx)
            local_loss.clear()
            class_loss.clear()
            start_time = datetime.datetime.now()

        del(image_tensors)
        gc.collect()
        
    print('Done fine-tuning!')


# Set up forward + backward pass for a single train step.
def get_model_train_step_function(model, optimizer, vars_to_fine_tune,batch_size,detection_model):
    @tf.function
    def train_step_fn(image_tensors,
                    groundtruth_boxes_list,
                    groundtruth_classes_list):
        """A single training iteration.

        Args:
        image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
            Note that the height and width can vary across images, as they are
            reshaped within this function to be 640x640.
        groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
            tf.float32 representing groundtruth boxes for each image in the batch.
        groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
            with type tf.float32 representing groundtruth boxes for each image in
            the batch.

        Returns:
        A scalar tensor representing the total loss for the input batch.
        """
        shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
        model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list,groundtruth_classes_list=groundtruth_classes_list)
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat([detection_model.preprocess(image_tensor)[0] for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        return [losses_dict['Loss/localization_loss'], losses_dict['Loss/classification_loss'], prediction_dict]
    return train_step_fn

def ReadLabelMap(path):
    label_map = label_map_util.create_category_index_from_labelmap(path,use_display_name=True)
    #categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=label_map_util.get_max_label_map_index(label_map),use_display_name=True)
    #category_index = label_map_util.create_category_index(categories)

    return label_map

 
def InferRandomImage(image_tensors,detection_model,prediction_dict,batch_size,fileName,idx):
    shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
    detections =  detection_model.postprocess(prediction_dict, shapes)
    image_np = helper.ConvertFiletoNumpy("./labelled_data/"+fileName)
    
    image_np_with_detections = viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + 1).astype(int),  #Add one is mandatory as 1st class is always background
            detections['detection_scores'][0].numpy(),
            inference.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=25,
            min_score_thresh=.10)
            
    image_np_with_detections = cv2.cvtColor(image_np_with_detections,cv2.COLOR_BGR2RGB)
    
    #cv2.imshow('Started Recording...',image_np_with_detections[0].numpy())
    cv2.imwrite("./videos/"+str(idx)+".jpg",image_np_with_detections)
    del(shapes)
    del(detections)
    del(image_np)
    del(image_np_with_detections)

'''
def GetAccuracy(model,train_images_np,batch_size,prediction_dict):

    shapes = train_images_np.shape
    all_keys = list(range(len(train_images_np)))
    random.shuffle(all_keys)
    example_keys = all_keys[:batch_size]

    gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
    gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
    
    image_tensors = [train_image_tensors[key] for key in example_keys]
    model.provide_groundtruth(groundtruth_boxes_list=gt_boxes_list,groundtruth_classes_list=gt_classes_list)

    preprocessed_images = tf.concat([model.preprocess(image_tensor)[0] for image_tensor in image_tensors], axis=0)
    #prediction_dict = model.predict(preprocessed_images, shapes)
    losses_dict = model.loss(prediction_dict, shapes)
    total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
    print(total_loss)
            
def Test(detection_model,learning_rate,batch_size):
    trainable_variables = detection_model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = ['WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
    'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
        
    
'''
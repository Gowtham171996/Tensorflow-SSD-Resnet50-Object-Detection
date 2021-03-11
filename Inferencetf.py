
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from object_detection.utils import visualization_utils as viz_utils


def plot_detections(image_np,
                    boxes,
                    classes,
                    scores,
                    category_index,
                    figsize=(12, 16),
                    image_name=None):
  """Wrapper function to visualize detections.

  Args:
    image_np: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    figsize: size for the figure.
    image_name: a name for the image file.
  """
  image_np_with_annotations = image_np.copy()
  viz_utils.visualize_boxes_and_labels_on_image_array(
      image_np_with_annotations,
      boxes,
      classes,
      scores,
      category_index,
      use_normalized_coordinates=True,
      min_score_thresh=0.01)
  if image_name:
    plt.imsave(image_name, image_np_with_annotations)
    #cv2.imshow("hello",image_np_with_annotations)
  else:
    plt.imshow(image_np_with_annotations)


@tf.function
def detect(input_tensor,detection_model):
  """Run detection on an input image.

  Args:
  input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

  Returns:
  A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
  """

  #input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
  preprocessed_image, shapes = detection_model.preprocess(input_tensor)
  prediction_dict = detection_model.predict(preprocessed_image, shapes)
  post_processed = detection_model.postprocess(prediction_dict, shapes)
  return post_processed,prediction_dict


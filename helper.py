from PIL import Image
import tensorflow as tf
import numpy as np
from six import BytesIO
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tabulate import tabulate

class CustomTensors:
	def __init__(self,filename, width, height,classname, xmin, ymin, xmax, ymax):
		self.filename = filename
		self.width = width
		self.height = height
		self.classname = classname
		self.xmin = xmin
		self.xmax = xmax
		self.ymin = ymin
		self.ymax = ymax
		#Required normalised coordinates for training from range 0 to 1
		self.gtboxNP = np.array([[ymin/height,xmin/width,ymax/height,xmax/width]], dtype=np.float32)
		#self.imageNP = [x[1] for x in train_images_np if x[0] == self.filename]

#img_data = tf.io.gfile.GFile(img_file_path, 'rb').read()
#image = Image.open(BytesIO(img_data))
	
def ConvertFiletoNumpy(img_file_path):
	image = Image.open(img_file_path)
	image.load() 
	(im_width, im_height) = image.size
	background = Image.new("RGB", image.size, (255, 255, 255))
	background.paste(image, mask=image.split()[3])
	return np.array(background.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  
def load_image_into_numpy_array(path):
	train_images_np = []
	k = 0
	for img_file in glob.glob(path + '/*.png'):
		k = k + 1
		train_images_np.append([img_file.split('/')[len(img_file.split('/'))-1],ConvertFiletoNumpy(img_file)])
		if(k % 100 == 0):
			print("Finished reading "+ str(k) + " images.")
		#if(k == 20):
		#	print("Forced stop reading at "+ str(k) + " images.")
		#	break
	return train_images_np

def xml_to_CustomTensorArray_file(xml_file):
	xml_object_list = []
	tree = minidom.parse(xml_file)
	for member in tree.getElementsByTagName('object'):
		value = CustomTensors( tree.getElementsByTagName('filename')[0].childNodes[0].data, #filename
                     int(tree.getElementsByTagName('width')[0].childNodes[0].data),   #width 
                     int(tree.getElementsByTagName('height')[0].childNodes[0].data),   #height 
                     member.childNodes[0].childNodes[0].data,                   #class
                     float(member.childNodes[1].childNodes[0].childNodes[0].data),           #xmin
                     float(member.childNodes[1].childNodes[1].childNodes[0].data),           #ymin
                     float(member.childNodes[1].childNodes[2].childNodes[0].data),           #xmax
                     float(member.childNodes[1].childNodes[3].childNodes[0].data),           #ymax
                     )
		xml_object_list.append(value)
	return xml_object_list


def xml_to_CustomTensorArray_folder(path):
	xml_list = []
	for xml_file in glob.glob(path + '/*.xml'):
		xml_list.extend(xml_to_CustomTensorArray_file(xml_file))
	return xml_list

	#					0			1		2		3		4		5		6		7
    #column_name = ['filename', 'width', 'height','class', 'xmin', 'ymin', 'xmax', 'ymax']
    #xml_df = pd.DataFrame(xml_list, columns=column_name)
    #return xml_df

def FileRead(file_path):
	f=open(file_path, "r")
	contents = f.readlines()
	f.close()
	i = 0
	category_index = {}
	for content in contents:
		i = i + 1
		category_index[content.replace('\n','')] = i
	return category_index

def AnnotationsSummary(category_index,getXMLAnnotationsImagesNames):
	#table = [["Sun",696000,1989100000],["Earth",6371,5973.6],
    #        ["Moon",1737,73.5],["Mars",3390,641.85]]
	#print(tabulate(table))
	class_distributions = []
	for row in category_index:
		count = len(list(filter(lambda x: str(x.classname).lower() == str(row).lower(), getXMLAnnotationsImagesNames)))
		class_distribution = [row, count,float((count/len(getXMLAnnotationsImagesNames))*100)]
		class_distributions.append(class_distribution)
	#class_distributions.sort(key=lambda x: x.count)
	print(tabulate(class_distributions,headers=["Class Name","Frequency count", "'%' out of 100"],tablefmt="fancy_grid"))
	print("Total Tensors count:" + str(len(getXMLAnnotationsImagesNames)))


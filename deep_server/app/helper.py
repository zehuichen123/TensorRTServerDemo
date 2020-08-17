import numpy as np
from PIL import Image
import base64
import requests
import io
import time
import datetime
import os
import matplotlib.pyplot as plt
import cv2

cate_list = []
with open('data/class_labels.txt', 'r') as f:
	lines = f.readlines()
	for line in lines:
		cate_list.append(line.strip())
# print(cate_list[:10])

def get_model_version():
	"""
	Usage:
	Get the current model version
	"""
	return 1

def save_file(img):
	"""
	Usage:
	Save img in local storage and return the img path
	"""
	ts = datetime.datetime.now()
	year, month, date, hour = ts.year, ts.month, ts.day, ts.hour
	rand_num = np.random.randint(0, 1e10)
	folder_path = 'data/%s/%s/%s/%s/' % (year, month, date, hour)
	os.makedirs(folder_path, exist_ok=True)
	img_name = '%s.jpg' % rand_num
	save_path = folder_path + img_name
	img = img.convert("RGB")
	img.save(save_path)
	return save_path

def parse_file(file):
	"""
	Usage:
	parse image before send it to GPU server and database
	currently resize img to 224 x 224
	"""
	img = Image.open(file)
	img = img.resize((224, 224))
	img_path = save_file(img)

	data = np.array(img)
	data = data / 255.0
	mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
	std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
	data = (data - mean) / std
	data = data.transpose(2, 0, 1)

	return data.astype(np.float32), img_path

def request_gpu_result(runner, data):
	"""
	Usage:
	Request to GPU server for inference results
	return: softmax vector
	"""
	results = runner.run(input={"input.0": data})
	return results['identity_output.0']

def get_cls_category(result):
	"""
	Usage:
	Get results with argmax to find the most 
	possible category predicted by model
	"""
	result = np.array(result[0])
	arg_index = np.argmax(result)
	print(arg_index)
	return cate_list[arg_index], arg_index

def save_results(record_template, result, img_path):
	"""
	Usage:
	Save results to MySQL
	"""
	ts = time.time()
	timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
	record = record_template(timestamp=timestamp, model_id=get_model_version(), \
							img_path=img_path, category=int(result))
	record.add()
	
	

	


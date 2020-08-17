from app import app
from flask import request
from flask import Response
from app.models import Record

import numpy as np
import os

from app import helper

from trt_client import client

runner = client.Inference(
	url="localhost:8001", # grpc
	model_name="plan_model",
	model_version=1,
)

@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"

@app.route("/api/cls", methods=['POST'])
def parse_request():
    # receive images from clients
    file = request.files["file"]

    # parse images before sending it to GPU server
    data, img_path = helper.parse_file(file)     # (224,224,3)

    # send to GPU server
    results = helper.request_gpu_result(runner, data)

    # mapping results to encoded category
    pred_cate, pred_index = helper.get_cls_category(results)

    # insert results to mysql database
    helper.save_results(record_template=Record, result=pred_index, img_path=img_path)

    return pred_cate, 200

@app.route("/api/cls_gpu", methods=['POST'])
def classify():
    return "Success!", 200
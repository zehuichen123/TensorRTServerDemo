sudo docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -v /home/czh/code/demo_model/triton_model/:/models nvcr.io/nvidia/tritonserver:20.03-py3 trtserver --model-store=/models

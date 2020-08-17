import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

class ModelData(object):
    INPUT_SHAPE = (3, 224, 224)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
engine_path = 'save_model/sample.engine'

def load_model(engine_path):
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)),\
                                                    dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)),\
                                                    dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async(bindings=[int(d_input), int(d_output)], \
                                    stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = ModelData.INPUT_SHAPE
        image_arr = cv2.resize(image, (h, w)).transpose(2, 0, 1)
        image_arr=image_arr.astype(trt.nptype(ModelData.DTYPE)).flatten()
        return (image_arr / 255.0 - 0.45) / 0.225

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(cv2.imread(test_image)))
    return test_image

engine = load_model(engine_path)
h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
with engine.create_execution_context() as context:
    checkpoint = time.time()
    for ii in range(100):
        test_image = '/home/czh/code/others/cat2.jpg'
        test_case = load_normalized_test_case(test_image, h_input)
        do_inference(context, h_input, d_input, h_output, d_output, stream)
    print("Per Img costs %g s" % ((time.time() - checkpoint) / 100))
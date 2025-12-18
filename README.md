# EC500 A2 AI Methods In Photonics
## Programming an Edge-computing GPU for Optical Experiments


For this class final project, I ran several AI models on the NVIDIA Jetson Nano. To do so, I wrote alexnet.py to run image classification using the pretrained AlexNet model, and the yolov8_trt*.py programs (which also support YOLO v5 models despite the naming) to convert the YOLO models to supported and optimized versions for the GPU.

YOLO executed with double precision is executed by the yolov8_trt.py program, and single precision is executed by yolov8_trtfp16.py, whil INT8 quantization is executed by  yolov8_trtint8.py.

from transformers import pipeline
import cv2
# from utils import *
from timm.models import create_model
from timm.data import resolve_data_config
import torch.nn as nn
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global config

    model = create_model(
                'efficientnet_b0',
                num_classes=2,
                in_chans=3,
                pretrained=True,
                checkpoint_path='best_models/efficientnet_b0/model_best.pth.tar' # maybe download the model from colab
                    )
    model = model.cuda()
    model.eval()

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global config
    # here we need to download the image and save it somewhere 
    # so we can read it via torch and insert it in cuda

    img = cv2.imread('01a51d500097.png')
    with torch.no_grad():
        
        img = torch.from_numpy(img).float().to('cuda')
        img = img.permute(2, 0, 1)
        img.unsqueeze_(0)
        labels = model(img)
        # topk = labels.topk(k)[1]
        label = labels.topk(1)[1].item()

    return {'result':label}

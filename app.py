
import os
import shutil
import torch
import requests
from timm.models import create_model
from timm.data import ImageDataset, create_loader, resolve_data_config

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global config

    url = 'https://impulse.ai/models/model_best.pth.tar'
    models_dir = 'models'

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists("models/model_best.pth.tar"):
        download_file(url, models_dir)

    model = create_model('resnet50', num_classes=2, in_chans=3, pretrained=True, checkpoint_path="models/model_best.pth.tar") # maybe download the model from colab

    model = model.cuda()
    model.eval()

    config = resolve_data_config({}, model=model)
    print(config)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    classes = ['cancer', 'not cancer']
    config = resolve_data_config({}, model=model)
    # Parse out your arguments
    img_url = model_inputs.get('image', None)
    if img_url == None:
        return {'message': "No image provided"}
    print("Downloading Images")
    img_folder = dl_img(img_url)
    
    print("creating Dataloader")
    loader = load_ds(config, test_path=img_folder) 

    with torch.no_grad():
        for batch_idx, (input, aw) in enumerate(loader):
            input = input.cuda()
            # Run the model
            labels = model(input)
            result = labels.topk(1)[1][0].item()

    # Return the results as a dictionary
    return {'result':classes[result]}

def load_ds(config, test_path='images', num_workers=2, batch_size=8):
    
    print(config)
    loader = create_loader(
        ImageDataset(test_path),
        input_size=config['input_size'],
        batch_size=batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=num_workers,
        crop_pct=config['crop_pct'])

    return loader

def dl_img(img, fname='images'):
    if not os.path.exists(fname):
        os.makedirs(fname)
    else:
        shutil.rmtree(fname)
        os.makedirs(fname)
    download_file(img, fname)
    return fname

def download_file(url, dir):

    local_filename = os.path.join(dir, url.split('/')[-1])

    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename

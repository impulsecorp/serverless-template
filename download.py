# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

import os
import requests
import shutil


def download_model():
    url = 'https://impulse.ai/models/model_best.pth.tar'
    models_dir = 'models'

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    download_file(url, models_dir)    
    
    # do a dry run of loading the huggingface model, which will download weights
    # pipeline('fill-mask', model='bert-base-uncased')

def download_file(url, models_dir):

    local_filename = os.path.join(models_dir, url.split('/')[-1])

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

if __name__ == "__main__":
    download_model()
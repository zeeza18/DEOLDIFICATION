DEOLDIFICATION BY MOHAMMED AZEEZULLA

CLONING WITH GIT'S DEOLDIFY REPOSITORY
"""

!git clone https://github.com/jantic/DeOldify.git DeOldify

"""CREATING A DIRECTORY TO STORE FILES"""

# Commented out IPython magic to ensure Python compatibility.
# %cd DeOldify

"""IMPORTING AND CHECKING FOR GPU RUNTIME"""

from deoldify import device
from deoldify.device_id import DeviceId
device.set(device=DeviceId.GPU0)
import torch
if not torch.cuda.is_available():
  print('GPU not available.')
from os import path

"""INSTALLING COLAB REUIREMENTS"""

!pip install -r colab_requirements.txt

import fastai
from deoldify.visualize import *
from pathlib import Path
torch.backends.cudnn.benchmark=True
import warnings
warnings.filterwarnings ("ignore", category=UserWarning, message="You've empty set")

"""ACCESSING COLORIZERS FOR IMAGE'S"""

!mkdir 'models'
!wget https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth -O ./models/ColorizeArtistic_gen.pth

"""WATERMARKS FOR IMAGES"""

!wget https://media.githubusercontent.com/media/jantic/DeOldify/master/resource_images/watermark.png -O ./resource_images/watermark.png

"""CALLING THE IMAGE OBJECT"""

colorizer_image = get_image_colorizer(artistic=True)

"""ACCESSING COLORIZER FOR VIDEO'S"""

!mkdir 'models'
!wget https://data.deepai.org/deoldify/ColorizeVideo_gen.pth -O ./models/ColorizeVideo_gen.pth

"""CALLING VIDEO OBJECT"""

# create object
colorizer_video = get_video_colorizer()

"""GUI FOR IMAGE"""

source_url = 'https://live.staticflickr.com/65535/48895826501_08d6f732c7_z.jpg' #@param {type:"string"}
render_factor = 7  #@param {type: "slider", min: 7, max: 40}
watermarked = True #@param {type:"boolean"}

if source_url is not None and source_url !='':
    image_path = colorizer_image.plot_transformed_image_from_url(url=source_url, render_factor=render_factor, compare=True, watermarked=watermarked)
    show_image_in_notebook(image_path)
else:
    print('Provide an image url and try again.')

"""GUI FOR VIDEO"""

# download video colorizing model
source_url = 'https://www.youtube.com/watch?v=4ZXCiKqAzoc' #@param {type:"string"}
render_factor = 5  #@param {type: "slider", min: 5, max: 40}
watermarked = False #@param {type:"boolean"}

if source_url is not None and source_url !='':
    video_path = colorizer_video.colorize_from_url(source_url, 'video.mp4', render_factor, watermarked=watermarked)
    show_video_in_notebook(video_path)
else:
    print('Provide a video url and try again.')
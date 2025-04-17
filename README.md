## this is a segmentation point cloud by long_term trajectories loss
this original code is Guess what move
## reproduct report:

this have been test in pytorch 2.6, python 3.12 and cuda 12 
```bash
python3.12 -m venv venv
source venv/bin/activate 
```
```bash
pip install torch torchvision torchaudio
pip install kornia jupyter tensorboard timm einops scikit-learn scikit-image openexr-python tqdm fontconfig
pip install cvbase opencv-python wandb
```

to install detectron2 from source, we need to use this command, remember to add `--no-build-isolation`
```bash
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
```

#### Data Preparation

Datasets should be placed under `data/<dataset_name>`, e.g. `data/DAVIS2016`.

* For video segmentation we follow the dataset preparation steps of [MotionGrouping](https://github.com/charigyang/motiongrouping).
* For image segmentation we follow the dataset preparation steps of [unsupervised-image-segmentation](https://github.com/lukemelas/unsupervised-image-segmentation).

#### code for generate cotracker
```python
import os
import torch
import numpy as np
import cv2
from base64 import b64encode
from IPython.display import HTML
from cotracker.predictor import CoTrackerPredictor
from tqdm import tqdm
def read_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return np.array(images)

def show_video(video_path):
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width="640" height="480" autoplay loop controls><source src="{video_url}"></video>""")

def process_videos(input_dir, output_dir, model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir in tqdm(os.listdir(input_dir)):
        subdir_path = os.path.join(input_dir, subdir)
        if os.path.isdir(subdir_path):
            video = read_images_from_folder(subdir_path)
            video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

            if torch.cuda.is_available():
                model = model.cuda()
                video = video.cuda()

            pred_tracks, pred_visibility = model(video, grid_size=30)
            print(f"Processed {subdir}: {pred_tracks.shape}, {pred_visibility.shape}")

            output_path_tracks = os.path.join(output_dir, f"{subdir}_tracks.npy")
            np.save(output_path_tracks, pred_tracks.cpu().numpy())
            output_path_visibility = os.path.join(output_dir, f"{subdir}_visibility.npy")
            np.save(output_path_visibility, pred_visibility.cpu().numpy())

model = CoTrackerPredictor(
    checkpoint=os.path.join(
        '/workspace/co-tracker/checkpoints/scaled_offline.pth'
    )
)

input_dir = '/workspace/guess-what-moves/DAVIS2016/JPEGImages/480p/'
output_dir = '/workspace/guess-what-moves/DAVIS2016/Traj/480p/'

process_videos(input_dir, output_dir, model)

```

### Running

#### Training

Experiments are controlled through a mix of config files and command line arguments. See config files and [`src/config.py`](src/config.py) for a list of all available options.

```bash
python main.py
```
Run the above commands in [`src`](src) folder.

#### Evaluation

Evaluation scripts are provided as [`eval-vid_segmentation.ipynb`](src/eval-vid_segmentation.ipynb) and [`eval-img_segmentation.ipynb`](src/eval-img_segmentation.ipynb) notebooks.



### Acknowledgements

This repository builds on [MaskFormer](https://github.com/facebookresearch/MaskFormer), [MotionGrouping](https://github.com/charigyang/motiongrouping), [unsupervised-image-segmentation](https://github.com/lukemelas/unsupervised-image-segmentation), [dino-vit-features](https://github.com/ShirAmir/dino-vit-features), and especially, [guess-what-move](https://github.com/karazijal/guess-what-moves).


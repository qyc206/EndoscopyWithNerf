# 3D Reconstruction of Endoscopy Images with NeRF

## Abstract

During endoscopic procedures, a camera is inserted into the human body to capture images of an inner organ. However, with a single scan of the organ, the camera is unable to capture all of the images needed to provide a complete assessment of the situation in the body. As a result, polyps, or protrusions on the surface of the examined organ, could be missed during the flythrough. The various factors that contribute to polyp misses include camera blind spots, haustral folds hiding polyps, and the inability of the camera angle to capture the polyp shape. Previous works that attempt to lower the polyp miss rate focus on improving the segmentation of the haustral folds in hopes of guiding the camera to areas where polyps could be hidden. The goal of this research, however, is to use Neural Radiance Fields (NeRF), a neural rendering model, to fill in the gaps of missing views in captured endoscopy images and reconstruct a scene that provides information to assist radiologists in identifying anomalies. When trained on a set of real stomach images, the NeRF reconstruction of the scene reveals a protrusion that is unidentifiable when observing the set of images alone. This suggests the potential in using this technique to recover polyps that are lost with the missed or uncaptured views.

[Google Slide Presentation](https://docs.google.com/presentation/d/14XVHCLMbdZn7RuGTQMlrS_MPJw3D9bfCjqDKJ1hALdA/edit?usp=sharing)

[Thesis Paper](https://github.com/qyc206/EndoscopyWithNerf/blob/main/qyc206_thesis_paper.pdf)

## About the dataset

The endoscopy images that were used to train NeRF model were from the [EndoSLAM dataset](https://github.com/CapsuleEndoscope/EndoSLAM), specifically from Cameras -> HighCam -> Stomach-III -> TumorfreeTrajectory_4. The folder contains the frames and an excel file with the camera pose information. Some notes on EndoSLAM camera information can be found via [this Notion link](https://www.notion.so/EndoSLAM-Camera-Info-310e55c0e026482b8ad3cc2735b674c6).

For this research, 20 consecutive stomach frames with their corresponding camera poses were used to train NeRF. 

## Where to start

Use the [train_test_nerf/convert2npy.py](https://github.com/qyc206/EndoscopyWithNerf/blob/main/train_test_nerf/convert2npy.py) to prepare the poses_bounds.npy file that is needed for training. Once this file is obtained, place it along with the "images" folder containing the corresponding frames in a folder for training. 

Use this [train_test_nerf/train_nerf_colab.ipynb](https://github.com/qyc206/EndoscopyWithNerf/blob/main/train_test_nerf/train_nerf_colab.ipynb) ([colab version](https://colab.research.google.com/drive/1FI1iOV0Z5kV9qNJBPa8vTElEFnh6jVVn?usp=sharing)) notebook for training and testing a NeRF model with custom data. This notebook also contains some code for obtaining and visualizing the obtained depth predictions. 

Follow the [README](https://github.com/qyc206/EndoscopyWithNerf/blob/main/visualize_cameras/README.md) in the [visualize_cameras](https://github.com/qyc206/EndoscopyWithNerf/tree/main/visualize_cameras) folder to visualize the camera poses from the poses_bounds.npy file.

## Extras

If you would like to render your own frames (i.e. obtain a synthetic dataset) via Blender, use the code files in [render_custom_frames_blender](https://github.com/qyc206/EndoscopyWithNerf/tree/main/render_custom_frames_blender) folder. 

This [google drive folder](https://drive.google.com/drive/folders/19M-abwXits9HcVM82lurrhMnRVqqY8df?usp=sharing) contains the results from the trials and tests that I have ran. 
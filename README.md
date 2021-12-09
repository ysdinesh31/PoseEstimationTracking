# Human Pose Estimation and Tracking
**This research/code is cited from: (Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, Yaser Sheikh, OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields) & (Steven Chen, Richard R. Yang, Pose Trainer: Correcting Exercise Posture using Pose Estimation)**

## Introduction
In this project uses the concept of OpenPose to detect and track human body by generating keypoints on body joints and then connecting them forming a skeleton.This project was later extended to track a person performing the bicep curls workout and count the repitions only when the form of the exercise is proper. This projects uses the model developed by CMU <a href = 'https://github.com/CMU-Perceptual-Computing-Lab/openpose'>Carnegie Mellon University</a>.

## Directory Structure

## Python files
all the files are obtained from https://github.com/akhilvasvani/StackGAN-v2/tree/master/code except generate_images.py which is our built in code.
There are many files to list but here are the imported ones
  - **bicepcurl.py** main file that tracks and counts the bicep curl
  - **OpenPose_Notebook.py** performs basic pose estimation tasks
## How to run code
1) First download the model files from https://github.com/CMU-Perceptual-Computing-Lab/openpose or use the following command to clone the entire repository: gh repo clone CMU-Perceptual-Computing-Lab/openpose
2) Save all the files as mentioned in the directory structure and then run the code using the following commands

## Dependencies for executing

python 3.6+
Visual Studio Code or any other editor
OpenCV
Mediapipe 
NumPy




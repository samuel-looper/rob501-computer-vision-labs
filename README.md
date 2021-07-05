# ROB501 Labs
Lab work for ROB501 (Computer Vision for Robotics). Project code is written in Python, and some code skeletons and helper functions written by the ROB501 teaching team.

Lab #1: Applied Image Homography
- main
  - dlt_homography.py:  Finds optimal coordinate transform between two images based on point correspondences using the DLT homography algorithm.
  - bilinear_interp.py: Rasterizes image warped by a homography by performing bilinear interpolation on pixels.
  - alpha_blend.py:     Overlays warped image onto background image with a degree of translucency using alpha blending.
  - terrain_overlay.py: Uses image homography and alpha blending to display terrain obstacle map onto input camera imagery from a mobile rover.
  - part_01_learner_example.py, part_02_learner_example.py, part_03_learner_example.py: Examples of code to evaluate algorithm
- images: data to test algorithms in learner examples 

Lab #2: Camera Calibration
- main:
  - saddle_point.py:        Finds saddle points in a grayscale image pixel map
  - cross_junctions.py:     Finds cross junctions corresponding to points on a calibration check board
  - find_jacobian.py:       Calculates the Jacobian of the pinhole camera model with respect to a single 3D landmark point
  - pose_estimate_nls.py:   Performs nonlinear least-squares optimization to find optimal coordinate transform to a calibration target
  - part_01_learner_example.py, part_02_learner_example.py, part_03_learner_example.py, part_04_learner_example.py: Examples of code to evaluate algorithm
- targets:   test images of calibration boards for learner examples 
- data:      stores intermediate array data and images

Lab #3: Stereo Image Matching
- main:
  - stereo_disparity_best.py:   Most accurate implementation of stereo disparity mapping using stereo rectification and standard kernel filters
  - stereo_disparity_score.py:  Helper function used to calculate accuracy of disparity map  
  - part_01_learner_example.py, part_02_learner_example.py: Examples of code to evaluate algorithm
- stereo: test stereo images and disparity maps for learner examples
- data:   stores intermediate array data and images

Lab #4: Visual Odometry
- main:
  - triangulate.py:       Computes optimal 3D position estimate of a given image feature given two rectified stereo images
  - estimate_motion_ls:   Scalar weighted method to find initial 3D pose estimate for visual odometry
  - estimate_motion_ils:  Performs nonlinear least-squares optimziation to find optimal 3D pose estimate for visual odometry
  - part_01_learner_example.py, part_02_learner_example.py: Examples of code to evaluate algorithm
- support: includes helper functions for coordinate transforms
- data:   stores intermediate array data and images

Lab #5: Sun Pose Estimation using Convolutional Neural Networks
- main:
  - test_sun_cnn.py: Evaluates CNN sun pose estimation models from supervised testing set
- models:
  -  sun_cnn.py:  Defines and trains a CNN sun pose estimation model in PyTorch

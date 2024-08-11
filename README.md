# StageAR: Markerless Mobile Phone Localization for AR in Live Events

## Hardware Components
- Calibrated cameras and LiDAR (intrinsic and extrinsic)
- Ouster OS-0, 128 beam LiDAR

## Software Environment
Python 3.10

Open3D 0.18.0

OpenCV 4.9

## Usage
### Static Model with External Camera Filtering
`/code/static_model.py`

### Dynamic Features from Fixed Stereo Depth
`/code/stereo_depth.py`

### LiDAR + Camera based Pose Estimation
`/code/lidar_mesh.py`

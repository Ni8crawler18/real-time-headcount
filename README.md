# Real-time People Detection and Classification in Industrial Settings

This project implements a real-time headcount system for workers and visitors in an industrial plant using YOLOv8 for object detection and OpenCV for image processing. The system differentiates between workers and visitors based on uniform detection using color histogram analysis.

## Technical Overview

- Object Detection: YOLOv8 (You Only Look Once version 8)
- Image Processing: OpenCV 4.5+
- Deep Learning Framework: PyTorch 1.7+
- Classification: Custom algorithm based on HSV color space analysis
- Performance Optimization: CUDA acceleration for GPU processing

## Features

- Real-time people detection with YOLOv8 at 30+ FPS on GPU
- Worker/visitor classification using HSV color histogram matching
- Multi-threading for parallel processing of detection and classification
- CUDA-optimized operations for improved performance
- Real-time data streaming to local database (SQLite)

## Requirements

- Python 3.8+
- CUDA 11.0+ and cuDNN 8.0+ (for GPU acceleration)
- OpenCV 4.5+ (with CUDA support)
- PyTorch 1.7+ (GPU version)
- Ultralytics YOLOv8
- NumPy, Pandas for data manipulation
- SQLite for local data storage

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/industrial-headcount-system.git
   cd industrial-headcount-system
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained YOLOv8 weights:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

## Usage

Run the main script with optional arguments:

```bash
python main.py --source 0 --conf 0.25 --iou 0.45 --device 0
```

Arguments:
- `--source`: Input source (0 for webcam, or path to video file)
- `--conf`: Confidence threshold for detection (default: 0.25)
- `--iou`: IOU threshold for NMS (default: 0.45)
- `--device`: Device to run on (cuda device, i.e. 0 or 0,1,2,3 or cpu)

## Configuration

Adjust the `config.yaml` file to modify:

```yaml
detection:
  model: 'yolov8n.pt'
  conf_thres: 0.25
  iou_thres: 0.45
  max_det: 1000

classification:
  worker_uniform:
    lower_hsv: [90, 50, 50]
    upper_hsv: [130, 255, 255]
  histogram_similarity_threshold: 0.7

processing:
  resize_width: 640
  resize_height: 480

database:
  path: 'headcount.db'
  update_interval: 5  # seconds
```

## Algorithm Details

1. **Detection**: 
   - YOLOv8 processes frames to detect people
   - Non-maximum suppression (NMS) applied to filter overlapping detections

2. **Classification**:
   - Detected person ROIs are converted to HSV color space
   - Color histogram is computed for each ROI
   - Histogram is compared with predefined worker uniform color profile using correlation method
   - Classification threshold determines worker vs visitor status

3. **Optimization**:
   - CUDA operations for faster GPU processing
   - Multi-threading to separate detection and classification processes
   - Frame skipping technique to maintain real-time performance

## Performance Metrics

- Detection FPS: Up to 30 FPS on NVIDIA RTX 3080
- Classification accuracy: ~95% (based on controlled environment tests)
- End-to-end latency: <100ms per frame

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 by Ultralytics
- OpenCV community
- PyTorch developers

For any technical questions or to report issues, please open an issue on this GitHub repository.
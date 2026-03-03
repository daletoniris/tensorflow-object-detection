# TensorFlow Object Detection on Raspberry Pi

Real-time object detection system built on TensorFlow Object Detection API, designed to run on Raspberry Pi and webcams. Includes a custom training pipeline for domain-specific models using SSD MobileNet.

\![TensorFlow](https://img.shields.io/badge/TensorFlow-1.x-orange?logo=tensorflow)
\![Python](https://img.shields.io/badge/Python-3.6+-blue?logo=python)
\![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-red?logo=raspberrypi)

## Features

- **Real-time detection** via webcam, video file, or PiCamera
- **Custom model training pipeline**: annotate → XML → CSV → TFRecord → train → freeze → deploy
- **SSD MobileNet v2** architecture for edge deployment
- **Pet detection alerts** with Twilio SMS integration
- **Autonomous robot control** via SSH commands on detection events

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  PiCamera / │────▶│  TF Object       │────▶│  Actions:   │
│  Webcam     │     │  Detection API   │     │  - SMS alert │
│  Video file │     │  (SSD MobileNet) │     │  - Robot cmd │
└─────────────┘     └──────────────────┘     │  - Logging   │
                                              └─────────────┘
```

## Training Pipeline

```bash
# 1. Annotate images with LabelImg → XML annotations
# 2. Convert XML annotations to CSV
python training_pipeline/xml_a_csv.py

# 3. Generate TFRecords for training
python training_pipeline/csv_a_tf.py

# 4. Train with TF Object Detection API
# 5. Export frozen inference graph
# 6. Deploy to Pi
python Object_detection_picamera.py
```

## Files

| File | Description |
|------|-------------|
| `detector.py` | Main detector with pet detection, SMS alerts, and robot control |
| `detector1.py` / `detector2.py` | Alternative detector configurations |
| `Object_detection_webcam.py` | Webcam-based detection |
| `Object_detection_video.py` | Video file detection |
| `Object_detection_picamera.py` | Raspberry Pi camera detection |
| `training_pipeline/xml_a_csv.py` | Convert XML annotations to CSV |
| `training_pipeline/csv_a_tf.py` | Generate TFRecords from CSV |
| `config/` | Label maps and model configuration |

## Requirements

```
tensorflow>=1.13
opencv-python
numpy
pillow
```

## Year

2018–2020

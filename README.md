# Police Car Cam
## Overview
Police Car Cam is an embedded system designed to assist law enforcement by providing automatic license plate recognition (ALPR) using artificial intelligence and image processing. The system enhances vehicle monitoring and law enforcement efficiency by reducing reliance on manual checks and enabling real-time detection of irregular vehicles.

https://github.com/user-attachments/assets/7fc2cd7b-dc6e-4a97-b623-4792a14b5203

## Features
- **Automatic License Plate Recognition (ALPR)**: Detects and recognizes vehicle license plates using convolutional neural networks (CNNs) based on the YOLOv3 model.
- **Real-time Processing**: Powered by a Neural Processing Unit (NPU) for fast inference without requiring cloud computing.
- **Embedded System**: Runs on Radxa ROCK 5C Lite, integrating a camera, GPS module, and LCD display.
- **Local and Remote Database Integration**: Syncs with a centralized database to check for irregular vehicles.
- **Alerts & Notifications**: Provides visual and auditory alerts for identified irregular vehicles.
- **On-Demand Video Recording**: Captures video and location data for future investigations.

https://github.com/user-attachments/assets/7a20699d-2f5d-4bd8-aba6-e7bdd41be9eb

## Technology Stack
- **Software**:
  - **YOLOv3-Tiny**: Object Detection Model
  - **Darknet**: Neural Network Framework
  - **RKNN Toolkit**: NPU Model Conversion Tool
  - **SQLite/PostgreSQL**: Database
  - **Python**: Programming Language
- **Hardware**:
  - **Radxa ROCK 5C Lite**: SBC
  - **Logitech C920e**: Camera
  - **NEO-6M**: GPS Module

## Project Highlights
### AI & Machine Learning:
- Training convolutional neural networks (CNNs) for license plate recognition.
- Optimizing deep learning models for embedded hardware.
- Implementing real-time object detection with YOLO.

### Embedded Systems:
- Running AI models on edge devices without cloud dependency.
- Integrating multiple hardware components into a single functional system.
- Using a Neural Processing Unit (NPU) for low-latency inference.

### Software Development:
- Designing a structured pipeline for image acquisition, processing, and database interaction.
- Implementing asynchronous processes to handle real-time detection and alerting.
- Optimizing database queries for fast license plate lookups.

## Contributors
- Bruno Emanuel Zenatti
- Enzo Holzmann Gaio
- Felipe Stilner Eufranio
- Ian Massanobu Santos Ishikawa

## License
This project is licensed under the MIT License.


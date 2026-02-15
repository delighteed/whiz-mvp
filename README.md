# Whiz MVP — Real-Time Attention & Distraction Detection

Whiz MVP is a computer vision system that detects user attention drift and cognitive distraction in real time using only a standard webcam.

The system analyzes facial landmarks, eye behavior, head rotation, and contextual cues to estimate focus level and trigger feedback when sustained distraction is detected.

---

## Key Features

- Real-time webcam processing (OpenCV)
- Face landmark tracking (MediaPipe FaceMesh)
- Eye Aspect Ratio (EAR) for blink detection
- Head rotation (yaw) distraction detection
- Optional AI-based cognitive load estimation (Random Forest)
- On-screen real-time feedback panel
- Sound alerts and system notifications
- Multiple sensitivity modes (Relaxed → Driving)

---

## How It Works

1. Webcam frames are captured using OpenCV.
2. MediaPipe extracts facial landmarks.
3. Eye and head pose metrics are computed.
4. Distraction thresholds are evaluated over time.
5. Alerts are triggered if attention drift persists.

Optional AI calibration allows the system to train a lightweight classifier for personalized focus estimation.

---

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- pygame
- plyer

---

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run the application:

python main.py

---

## Project Purpose

Whiz MVP was developed as a proof-of-concept system to explore real-time, AI-driven attention and distraction detection using only consumer-grade hardware (a standard webcam).

The goal of this prototype is to demonstrate that cognitive state estimation can be performed locally, without wearable devices or cloud processing, using lightweight computer vision techniques.

This project serves as an early-stage validation of a broader AI-based digital wellbeing and productivity platform.

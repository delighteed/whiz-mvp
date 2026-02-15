# Whiz MVP — Real-Time Attention & Distraction Detection

Whiz MVP is a computer vision system that detects user attention drift and distraction in real time using only a standard webcam.

It analyzes facial landmarks, eye behavior, head rotation, and contextual cues to estimate focus level and trigger feedback when distraction persists.

---

## Key Features

- Real-time webcam processing (OpenCV)
- Face landmark tracking (MediaPipe)
- Eye Aspect Ratio (blink detection)
- Head rotation (yaw) distraction detection
- Optional AI-based load estimation (Random Forest)
- On-screen feedback panel
- Sound + system notification alerts
- Multiple sensitivity modes (Relaxed → Driving)

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

```bash
pip install -r requirements.txt

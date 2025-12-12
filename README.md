 Oracle Eye v2.6
**sees you before you see it**

A single-file, zero-dependency-after-install, biomimetic vision engine that runs at 60-75 FPS on a regular Windows laptop (tested on an old Navy i7 with integrated graphics).

It doesn’t just detect — it **anticipates**.  
When nothing moves, Oracle Eye starts “dreaming” plausible future frames and screams **“PROPHECY FAILURE — REALITY LIES!”** the moment the real world deviates from its prediction.

### Features
- Real-time webcam, video file, or full-screen capture  
- Works perfectly when motion is physically impossible (black screen, frozen feed, etc.)  
- Predictive “hallucination” mode that generates the next logical frame  
- Loud reality-check alerts when the future doesn’t match the prophecy  
- Zero GPU required  
- Zero internet required after the first run (models cached locally)  
- 100 % offline, 100 % Python, 100 % single file

### Requirements
Tested on Windows 10/11 • Python 3.9–3.12

```bash
pip install opencv-python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics mss pyautogui numpy
Run
Bashpython biomimetic_vision_v2.6.py
Built December 2025 by Cody Garrelts
Diploma-qualified ETV3

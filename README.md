# 🎮 Gesture-Control Gaming

Play PC games using hand gestures with Python, OpenCV, and MediaPipe.


---

## 🧠 About the Project

This project lets you control games like **Temple Run** or **Subway Surfers** using simple **hand gestures** instead of a keyboard.

It uses **MediaPipe** for hand tracking, **OpenCV** for webcam input, and **PyAutoGUI** to simulate keypresses.

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – real-time video processing
- **MediaPipe** – hand landmark/palm detection
- **PyAutoGUI** – keyboard emulation
- **NumPy**

---

## ⚙️ How It Works

- Detects full palm using your webcam
- Tracks hand position and movement
- Maps gestures (like hand up/down/side) to keyboard inputs
- Sends key events to control games

---

## 🖼️ Project Structure

gesture-game-controller/
├── main.py
├── requirements.txt
└── README.md


## ▶️ How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt

2.Run the code:
python main.py

3.Open your game (like Temple Run)

4.Use gestures to play!



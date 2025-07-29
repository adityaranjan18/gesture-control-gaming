import cv2
import mediapipe as mp
import pyautogui
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

prev_x, prev_y = 0, 0
threshold = 40
cooldown = 1.0
last_action_time = time.time()

palm_connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  
    (0, 5), (5, 6), (6, 7), (7, 8),  
    (5, 9), (9, 10), (10, 11), (11, 12),  
    (9, 13), (13, 14), (14, 15), (15, 16),  
    (13, 17), (17, 18), (18, 19), (19, 20),  
    (0, 17), (0, 13), (0, 9)  
]

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = image.shape

            for connection in palm_connections:
                start = hand_landmarks.landmark[connection[0]]
                end = hand_landmarks.landmark[connection[1]]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            palm_x = int(hand_landmarks.landmark[0].x * w)
            palm_y = int(hand_landmarks.landmark[0].y * h)
            cv2.circle(image, (palm_x, palm_y), 15, (0, 255, 0), -1)

            dx = palm_x - prev_x
            dy = palm_y - prev_y

            current_time = time.time()
            if current_time - last_action_time > cooldown:
                if abs(dx) > abs(dy):
                    if dx > threshold:
                        pyautogui.press('d')
                        last_action_time = current_time
                    elif dx < -threshold:
                        pyautogui.press('a')
                        last_action_time = current_time
                else:
                    if dy < -threshold:
                        pyautogui.press('w')
                        last_action_time = current_time
                    elif dy > threshold:
                        pyautogui.press('s')
                        last_action_time = current_time

            prev_x, prev_y = palm_x, palm_y

        cv2.imshow('Spidey Web Palm Control', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

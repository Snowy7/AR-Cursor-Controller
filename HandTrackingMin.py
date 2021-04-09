import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
mpDrawer = mp.solutions.drawing_utils
hands = mpHands.Hands()

pTime, cTime = 0, 0

while True:
    success, img = cap.read()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLM in results.multi_hand_landmarks:
            for id, lm in enumerate(handLM.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 0:
                    pass
                cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)

            mpDrawer.draw_landmarks(img, handLM, mpHands.HAND_CONNECTIONS)
    # fps part
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

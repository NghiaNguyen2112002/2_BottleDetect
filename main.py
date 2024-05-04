import cv2
import numpy as np
from time import time, strftime, gmtime

import os
import pygame


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

model = YOLO("../1_ObjectTracking/best.pt")
font = cv2.FONT_HERSHEY_DUPLEX
kamera = cv2.VideoCapture(0)



# Initialize the pygame mixer
pygame.mixer.init()

# Load a sound file
sound_file = "Hãy bỏ chai nước vào.mp3"
sound = pygame.mixer.Sound(sound_file)

region1 = np.array([(640, 0), (660, 0), (660, 720), (640, 720)])
region1 = region1.reshape((-1, 1, 2))
total = set()
milliseconds = 0
timeOutSpeaker = 0
flagSaved = 0

while True:
    print("Duration: ", int(time() * 1000) - milliseconds)
    milliseconds = int(time() * 1000)

    ret, frame = kamera.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    H, W, _ = frame.shape

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.track(rgb_img, persist=True, verbose=False)
    labels = results[0].names

    linePos = int(W * 4/5)

    cv2.line(frame, (linePos, 0), (linePos, H), (0, 0, 255), 2)

    # cv2.polylines(frame,[region1],True,(255,0,0),2)
    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i]
        score = results[0].boxes.conf[i]
        cls = results[0].boxes.cls[i]
        # ids = results[0].boxes.id[i]

        x1, y1, x2, y2, score, cls = int(x1), int(y1), int(x2), int(y2), float(score), int(cls)

        name = labels[cls]
        if name != 'bottle':
            continue
        if score < 0.5:
            continue

        if int(time() * 1000) - timeOutSpeaker > 5000:
            # Play the sound
            sound.play()
            timeOutSpeaker = int(time() * 1000)


        # Wait for the sound to finish playing
        # pygame.time.wait(int(sound.get_length() * 1000))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)
        # cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        cx = int(x1 / 2 + x2 / 2)
        cy = int(y1 / 2 + y2 / 2)

        if cx > linePos:
            cv2.circle(frame, (cx, cy), 15, (255, 0, 0), -1)
            if flagSaved == 0:
                flagSaved = 1
                name = "Học sinh vi phạm/" + strftime("%Y-%m-%d_%H-%M-%S", gmtime(time())) + ".jpg"
                print(name)
                cv2.imwrite(name, frame)     # save frame as JPEG file
        else:
            flagSaved = 0


        inside_region1 = cv2.pointPolygonTest(region1, (cx, cy), False)
        if inside_region1 > 0:
            cv2.line(frame, (linePos, 0), (linePos, H), (0, 255, 255), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
kamera.release()
cv2.destroyAllWindows()
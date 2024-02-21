import cv2
import os
import requests
from time import sleep, time


server_url = 'http://localhost:8080/recognize'

webcam = cv2.VideoCapture(0)
reading = True

previous_time = time()
delta = 0

while reading:
    reading, frame = webcam.read()
    cv2.imwrite('webcam_frame.jpg', frame)

    current_time = time()
    delta += current_time - previous_time
    previous_time = current_time

    if delta > 0:  # seconds
        image = open('webcam_frame.jpg', 'rb').read()
        try:
            response = requests.post(server_url, data=image).json()

            license_plate_text = response['license_plate_text']
            x1, y1, x2, y2 = response['license_plate_bbox']

            if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                cv2.putText(
                    frame, license_plate_text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3
                )
        except Exception as e:
            print(e)
            sleep(5)
            continue

        os.remove('webcam_frame.jpg')
        delta = 0

    cv2.imshow('Webcam test video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

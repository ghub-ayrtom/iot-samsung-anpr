import cv2
import numpy as np
import os
import requests
from time import sleep, time
import urllib.request


server_url = 'http://localhost:8080/recognize'
esp32cam_url = 'http://192.168.43.219/cam-hi.jpg'
esp32cam = cv2.VideoCapture(esp32cam_url)

if not esp32cam.isOpened():
    print("Failed to open the IP camera stream.")
    exit()

reading = True
previous_time = time()
delta = 0

while reading:
    # reading, frame = esp32cam.read()

    response = urllib.request.urlopen(esp32cam_url)
    image = np.array(bytearray(response.read()), dtype=np.uint8)

    image = cv2.imdecode(image, -1)
    # cv2.imwrite('esp32cam_frame.jpg', image)

    current_time = time()
    delta += current_time - previous_time
    previous_time = current_time

    if delta > 0:  # seconds
        # image = open('esp32cam_frame.jpg', 'rb').read()
        try:
            response = requests.post(server_url, data=image).json()

            license_plate_text = response['license_plate_text']
            x1, y1, x2, y2 = response['license_plate_bbox']

            if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                cv2.putText(
                    image, license_plate_text, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3
                )
        except Exception as e:
            print(e)
            sleep(5)
            continue

        os.remove('esp32cam_frame.jpg')
        delta = 0

    cv2.imshow('ESP32-CAM test video', image)
    if cv2.waitKey(1) == ord('q'):
        break

esp32cam.release()
cv2.destroyAllWindows()

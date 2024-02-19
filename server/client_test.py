import cv2
import os
import requests
from time import sleep, time

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

    if delta > 1:  # seconds
        url = 'http://localhost:8080/recognize'
        image = open('webcam_frame.jpg', 'rb').read()
        try:
            license_plate_text = requests.post(url, data=image)
            print(license_plate_text.text)
        except:
            print("Connection refused by the server.")
            sleep(5)
            continue

        os.remove('webcam_frame.jpg')
        delta = 0

    cv2.imshow('webcam_video', frame)
    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

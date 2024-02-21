import sys
sys.path.insert(1, '/home/ayrtom/PycharmProjects/iot-samsung-anpr/models/LPRNet')

import cv2
from models.LPRNet.Decoders import decode_function, BeamDecoder
from flask import Flask, request
import glob
from PIL import Image
import io
from models.LPRNet.LPRNet import load_default_lprnet
from models.LPRNet.SpatialTransformer import load_default_stn
import numpy as np
import os
from pytesseract import Output
import pytesseract
import torch
from ultralytics import YOLO

app = Flask(__name__)
upload_images_path = 'uploaded_images'
port = 8080

thresh_files = glob.glob('../thresh_set_up/*')
for file in thresh_files:
    os.remove(file)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

license_plate_detection_model = YOLO("../models/LPDM_pretrained.pt")
license_plate_recognition_model = load_default_lprnet(device)
spatial_transformer_model = load_default_stn(device)


def get_images_count(path):
    images_count = 1

    for image_file_name in os.listdir(path):
        if os.path.isfile(os.path.join(path, image_file_name)):
            images_count += 1

    return images_count


def convert_image_bytes_to_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    return image


def get_prediction(image_bytes, debug=True):
    # license_plates_database = ['B440TM73', '...']
    license_plate_image = convert_image_bytes_to_image(image_bytes)

    """
        conf: устанавливает минимальный порог уверенности для обнаружения (false positives reduce, 0.25 - default)
        iou: более высокие значения приводят к уменьшению количества обнаружений за счёт устранения перекрывающихся 
        боксов, что полезно для уменьшения количества дубликатов (multiply detections reduce, 0.7 - default)
        max_det: максимальное количество обнаружений, допустимое для одного изображения (300 - default)
    """
    license_plate_detections = license_plate_detection_model.predict(license_plate_image, conf=0.25, iou=0.7, max_det=300)

    for license_plate in license_plate_detections:
        for license_plate_bounding_box in license_plate.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = license_plate_bounding_box

            if debug:
                cv2.rectangle(license_plate_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

            license_plate_image_cropped = license_plate_image[int(y1):int(y2), int(x1):int(x2)]
            license_plate_image_cropped = cv2.resize(license_plate_image_cropped, (94, 24), interpolation=cv2.INTER_CUBIC)
            license_plate_image_cropped = (np.transpose(np.float32(license_plate_image_cropped),
                                                        (2, 0, 1)) - 127.5) * 0.0078125

            data = torch.from_numpy(license_plate_image_cropped).float().unsqueeze(0).to(device)

            license_plate_image_transformed = spatial_transformer_model(data)
            predictions = license_plate_recognition_model(license_plate_image_transformed)
            predictions = predictions.cpu().detach().numpy()

            labels, probability, predicted_labels = decode_function(
                predictions,
                [
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
                    'Y', 'X', '-',
                ],
                BeamDecoder
            )

            if (probability[0] < -85) and (len(labels[0]) in [8, 9]):
                # if labels[0] in license_plates_database:
                # return 'License plate number {} with {} probability.'.format(labels[0], probability[0])

                response = {
                    'license_plate_text': labels[0],
                    'license_plate_bbox': [x1, y1, x2, y2],
                }

                return response

            '''
    
            license_plate_cropped = license_plate_image[int(y1 + 10):int(y2 - 10), int(x1 + 5):int(x2 - 5)]
            # license_plate_resized = cv2.resize(license_plate_cropped, (200, 100))
            license_plate_grayscale = cv2.cvtColor(license_plate_cropped, cv2.COLOR_BGR2GRAY)
            # license_plate_blurry = cv2.medianBlur(license_plate_grayscale, 1)
            # license_plate_blurry = cv2.GaussianBlur(license_plate_grayscale, (1, 1), sigmaX=0)
            _, license_plate_thresh = cv2.threshold(license_plate_grayscale, 200, 255, cv2.THRESH_BINARY)
    
            kernel = np.ones((2, 2), np.uint8)
            license_plate_dilate = cv2.dilate(license_plate_thresh, kernel, iterations=1)  # white on black
    
            kernel = np.ones((3, 3), np.uint8)
            license_plate_erode = cv2.erode(license_plate_dilate, kernel, iterations=1)  # black on white
    
            if debug:
                thresh_images_count = get_images_count('../thresh_set_up')
                cv2.imwrite('../thresh_set_up/{}.jpg'.format(thresh_images_count), license_plate_erode)
    
            tesseract_recognized_text = pytesseract.image_to_data(
                license_plate_erode,
                lang='eng',
                config=f'--psm 7 --oem 3 -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789',
                output_type=Output.DICT,
            )
    
            for recognized_text in tesseract_recognized_text['text']:
                recognized_text = recognized_text.upper().replace(' ', '')
    
                if recognized_text in license_plates_database:
                    return recognized_text
                    
            '''

    response = {
        'license_plate_text': '',
        'license_plate_bbox': [-1, -1, -1, -1],
    }

    return response


@app.route('/recognize', methods=['POST'])
def recognize_license_plate(debug=False):
    if request.method == "POST":
        license_plate_image_raw_bytes = request.get_data()  # ESP32CAM

        if debug:
            uploaded_images_count = get_images_count(upload_images_path)
            save_location = (os.path.join(app.root_path, "{}/{}.jpg".format(upload_images_path, uploaded_images_count)))

            license_plate_image_file = open(save_location, 'wb')
            license_plate_image_file.write(license_plate_image_raw_bytes)
            license_plate_image_file.close()

        license_plate_text = get_prediction(license_plate_image_raw_bytes)
        return license_plate_text


@app.route("/")
def running_test():
    return "<p>The server is running!</p>"


if __name__ == '__main__':
    app.run(host='localhost', port=port)

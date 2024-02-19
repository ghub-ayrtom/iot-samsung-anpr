import cv2
import easyocr
import glob
import numpy as np
import os
from pytesseract import Output
import pytesseract
import requests
import string
from ultralytics import YOLO
from sort.sort import *

easyocr_reader = easyocr.Reader(['en'], gpu=False)

char_to_int = {
    'A': '4',
    'B': '8',
    'E': '3',
    'O': '0',
    'T': '7',
}

int_to_char = {
    '0': 'O',
    '3': 'E',
    '4': 'A',
    '7': 'T',
    '8': 'B',
}


def get_car_info(license_plate, detected_coco_objects_track_ids):
    x1_plate, y1_plate, x2_plate, y2_plate, confidence, class_id = license_plate

    car_found = False
    car_index = -1

    for i in range(len(detected_coco_objects_track_ids)):
        x1_car, y1_car, x2_car, y2_car, car_id = detected_coco_objects_track_ids[i]

        if x1_plate > x1_car and y1_plate > y1_car and x2_plate < x2_car and y2_plate < y2_car:
            car_index = i
            car_found = True
            break

    if car_found and car_index != -1:
        return detected_coco_objects_track_ids[car_index]
    return -1, -1, -1, -1, -1


def check_license_plate_format(text):
    if len(text) != 8 or len(text) != 9:
        return False

    if len(text) == 8:
        if (text[0] in string.ascii_uppercase or text[0] in int_to_char.keys()) and \
                (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in char_to_int.keys()) and \
                (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in char_to_int.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in char_to_int.keys()) and \
                (text[4] in string.ascii_uppercase or text[4] in int_to_char.keys()) and \
                (text[5] in string.ascii_uppercase or text[5] in int_to_char.keys()) and \
                (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in char_to_int.keys()) and \
                (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in char_to_int.keys()):
            return True
        else:
            return False

    if len(text) == 9:
        if (text[0] in string.ascii_uppercase or text[0] in int_to_char.keys()) and \
                (text[1] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[1] in char_to_int.keys()) and \
                (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in char_to_int.keys()) and \
                (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in char_to_int.keys()) and \
                (text[4] in string.ascii_uppercase or text[4] in int_to_char.keys()) and \
                (text[5] in string.ascii_uppercase or text[5] in int_to_char.keys()) and \
                (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in char_to_int.keys()) and \
                (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in char_to_int.keys()) and \
                (text[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[8] in char_to_int.keys()):
            return True
        else:
            return False


def format_license_plate_text(text):
    license_plate_string = ''

    mapping = {
        0: int_to_char,
        1: char_to_int,
        2: char_to_int,
        3: char_to_int,
        4: int_to_char,
        5: int_to_char,
        6: char_to_int,
        7: char_to_int,
        8: char_to_int,
    }

    if len(text) == 8:
        for i in [0, 1, 2, 3, 4, 5, 6, 7]:
            if text[i] in mapping[i].keys():
                license_plate_string += mapping[i][text[i]]
            else:
                license_plate_string += text[i]

    if len(text) == 9:
        for i in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            if text[i] in mapping[i].keys():
                license_plate_string += mapping[i][text[i]]
            else:
                license_plate_string += text[i]

    return license_plate_string


def recognize_license_plate(license_plate_cropped):
    license_plates_database = ['B440TM73', '...']

    tesseract_recognized_text = pytesseract.image_to_data(
        license_plate_cropped,
        lang='eng',
        config=f'--psm 7 --oem 3 -c tessedit_char_whitelist=ABEKMHOPCTYX0123456789',
        output_type=Output.DICT,
    )

    for recognized_text in tesseract_recognized_text['text']:
        recognized_text = recognized_text.upper().replace(' ', '')

        if recognized_text in license_plates_database:
            return True, recognized_text

    '''
    
    easyocr_recognized_text = easyocr_reader.readtext(license_plate_cropped, allowlist='ABEKMHOPCTYX0123456789')

    for recognized_text in easyocr_recognized_text:
        bounding_box, text, confidence = license_plate_character
        text = text.upper().replace(' ', '')

        if text in license_plates_database:
            return True, text

        if check_license_plate_format(text):
            return format_license_plate_text(text), confidence
            
    '''

    return None, None


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate'][
                                                                'bbox_confidence'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate'][
                                                                'text_confidence'])
                            )
        f.close()

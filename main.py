from utils import *

pretrained_yolo_model = YOLO("models/yolov8n.pt")
license_plate_detection_model = YOLO("models/LPDM_pretrained.pt")

mot_tracker = Sort()
test_video = cv2.VideoCapture(0)  # 'data/test/russian_highway_traffic.mp4'
results = {}

frame_number = -1
file_count = 1
reading = True
detection_objects = [0, 2, 5, 7]  # coco.names (person, car, bus, truck)

thresh_files = glob.glob('thresh_set_up/*')
for file in thresh_files:
    os.remove(file)

while reading:
    frame_number += 1
    reading, frame = test_video.read()

    if reading:
        results[frame_number] = {}

        detections = pretrained_yolo_model(frame)[0]
        detected_coco_objects = []

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection

            if int(class_id) in detection_objects:
                detected_coco_objects.append([x1, y1, x2, y2, confidence])

        # track_ids = mot_tracker.update(np.asarray(detected_coco_objects))
        license_plates = license_plate_detection_model(frame)[0]

        for license_plate in license_plates.boxes.data.tolist():
            x1_plate, y1_plate, x2_plate, y2_plate, confidence, class_id = license_plate
            # x1_car, y1_car, x2_car, y2_car, car_id = get_car_info(license_plate, track_ids)

            cv2.rectangle(frame, (int(x1_plate), int(y1_plate)), (int(x2_plate), int(y2_plate)), (0, 0, 255), 5)

            license_plate_cropped = frame[int(y1_plate):int(y2_plate), int(x1_plate):int(x2_plate)]
            license_plate_grayscale = cv2.cvtColor(license_plate_cropped, cv2.COLOR_BGR2GRAY)
            # license_plate_blurry = cv2.medianBlur(license_plate_grayscale, 1)
            # license_plate_blurry = cv2.GaussianBlur(license_plate_grayscale, (1, 1), sigmaX=0)
            _, license_plate_thresh = cv2.threshold(license_plate_grayscale, 175, 255, cv2.THRESH_BINARY)

            kernel = np.ones((2, 2), np.uint8)
            license_plate_dilate = cv2.dilate(license_plate_thresh, kernel, iterations=1)  # white on black

            kernel = np.ones((3, 3), np.uint8)
            license_plate_erode = cv2.erode(license_plate_dilate, kernel, iterations=1)  # black on white

            cv2.imwrite('thresh_set_up/{}.jpg'.format(file_count), license_plate_erode)
            file_count += 1

            license_plate_is_recognized, license_plate_text = recognize_license_plate(license_plate_erode)

            if license_plate_is_recognized:
                print(license_plate_text)
                reading = False

            '''

            if car_id != -1:
                license_plate_cropped = frame[int(y1_plate):int(y2_plate), int(x1_plate):int(x2_plate)]
                license_plate_grayscale = cv2.cvtColor(license_plate_cropped, cv2.COLOR_BGR2GRAY)
                _, license_plate_thresh = cv2.threshold(license_plate_grayscale, 100, 255, cv2.THRESH_BINARY_INV)

                license_plate_text, license_plate_confidence = recognize_license_plate(license_plate_thresh)

                if license_plate_text is not None:
                    results[frame_number][car_id] = {
                        'car': {
                            'bbox': [x1_car, y1_car, x2_car, y2_car],
                        },
                        'license_plate': {
                            'bbox': [x1_plate, y1_plate, x2_plate, y2_plate],
                            'bbox_confidence': confidence,
                            'text': license_plate_text,
                            'text_confidence': license_plate_confidence
                        }
                    }
                    
            '''

        cv2.imshow('test_video', frame)
        if cv2.waitKey(1) == ord('q'):
            break

# write_csv(results, 'results.csv')

test_video.release()
cv2.destroyAllWindows()

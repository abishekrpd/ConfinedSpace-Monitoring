from ultralytics import YOLO
import cv2
import math 
import datetime
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("safesight\\best.pt")

# object classes
classNames = ['boots', 'face_mask', 'face_nomask', 'glasses', 'goggles',
                  'hand_glove', 'hand_noglove', 'head_helmet', 'head_nohelmet',
                  'person', 'shoes', 'vest']
fourcc = cv2.VideoWriter_fourcc(*'XVID')
is_recording=False
while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            img=cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            # print("Confidence --->",confidence)

            
            cls= classNames[int(box.cls.item())]
            
            # print("Class name -->", classNames[int(box.cls.item())])

            # # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img,cls, org, font, fontScale, color, thickness)
            if cls in ['hand_noglove', 'head_nohelmet', 'face_nomask'] and not is_recording:
                    is_recording = True
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_writer = cv2.VideoWriter(f'recording_{timestamp}.avi', fourcc, 20.0, (img.shape[1], img.shape[0]))
                    print("Start recording!")

            elif cls in ['hand_glove', 'head_helmet', 'face_mask'] and is_recording:
                    is_recording = False
                    video_writer.release()
                    print("Stop recording and save video!")


    cv2.imshow('Prediction', img)
    if is_recording:
            video_writer.write(img)
    
    if cv2.waitKey(1) == ord('q'):
        break

    # Release resources
if is_recording:
        video_writer.release()
cap.release()
cv2.destroyAllWindows()
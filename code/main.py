import cv2
import mediapipe as mp
import os
import argparse


def process_image(img,face_detection):
    H, W, _ = img.shape
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output_img = face_detection.process(rgb_img)

    if output_img.detections is not None:
        for detection in output_img.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)
            # img = cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),5)

            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (40, 40))
    return img


args = argparse.ArgumentParser()
args.add_argument("--mode",default='webcam')
args.add_argument("--filepath",default='None')
args = args.parse_args()


face_detected = mp.solutions.face_detection

with face_detected.FaceDetection(model_selection=0,min_detection_confidence = 0.5) as face_detection:
    if args.mode in ["image"]:
        img = cv2.imread(args.filepath)

        img = process_image(img,face_detection)
        cv2.imwrite('C:/Users/Srikar/PycharmProjects/face_anonymiser/data/my_face_anonymous.jpg',img)

    elif args.mode in["video"]:

        videoo = cv2.VideoCapture(args.filepath)
        ret,frame = videoo.read()

        output_vid = cv2.VideoWriter('C:/Users/Srikar/PycharmProjects/face_anonymiser/data/my_face_anonymous_video.mp4',
                                     cv2.VideoWriter_fourcc(*'MP4V'), 25, (frame.shape[1], frame.shape[0]))

        while ret:
            frame = process_image(frame, face_detection)
            output_vid.write(frame)
            ret,frame = videoo.read()

        videoo.release()
        output_vid.release()

    elif args.mode in ["webcam"]:

        livestream = cv2.VideoCapture(0)
        ret, frame = livestream.read()

        while ret:
            frame = process_image(frame, face_detection)
            cv2.imshow('frame',frame)
            cv2.waitKey(25)
            ret, frame = livestream.read()

        livestream.release()

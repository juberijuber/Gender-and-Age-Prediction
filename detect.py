import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog

# ---------------------- LOAD MODELS ----------------------
faceNet = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
ageNet = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male','Female']

# ---------------------- FACE DETECTION FUNCTION ----------------------
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300),
                                 [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3] * frameWidth)
            y1 = int(detections[0,0,i,4] * frameHeight)
            x2 = int(detections[0,0,i,5] * frameWidth)
            y2 = int(detections[0,0,i,6] * frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0),
                          int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

# ---------------------- IMAGE VERSION ----------------------
def detect_from_image():
    # Open file dialog
    root = tk.Tk()
    root.withdraw()  # hide main window
    image_path = filedialog.askopenfilename(title="Select an Image", 
                                            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not image_path:
        print("No file selected!")
        return

    image = cv2.imread(image_path)
    if image is None:
        print("Could not read image:", image_path)
        return
    
    resultImg, faceBoxes = highlightFace(faceNet, image)
    if not faceBoxes:
        print("No face detected")
    for faceBox in faceBoxes:
        face = image[max(0,faceBox[1]-20):min(faceBox[3]+20, image.shape[0]-1),
                     max(0,faceBox[0]-20):min(faceBox[2]+20, image.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        print(f"Gender: {gender}, Age: {age[1:-1]} years")
        cv2.putText(resultImg, f"{gender}, {age}", (faceBox[0], faceBox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Image Detection", resultImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------------------- WEBCAM VERSION ----------------------
def detect_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        for faceBox in faceBoxes:
            face = frame[max(0,faceBox[1]-20):min(faceBox[3]+20, frame.shape[0]-1),
                         max(0,faceBox[0]-20):min(faceBox[2]+20, frame.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            cv2.putText(resultImg, f"{gender}, {age}", (faceBox[0], faceBox[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Webcam Detection", resultImg)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------------- RUN ----------------------
if __name__ == "__main__":
    print("Choose Mode:\n1 - Upload Image\n2 - Use Webcam")
    mode = input("Enter choice (1/2): ").strip()

    if mode == "1":
        detect_from_image()
    elif mode == "2":
        detect_from_webcam()
    else:
        print("Invalid choice!")

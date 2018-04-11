import cv2
import os
import numpy as np

subjects = ["", "Ramiz Raja", "Elvis Presley"]


def face_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    # Assuming only one face
    (x, y, w, h) = faces[0]

    return gray[y:y + w, x:x + h], faces[0]


def prepare_training_data(training_data):
    dirs = os.listdir(training_data)
    # list for subjects
    faces = []
    # list for labels respectively
    labels = []

    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))
        subject_dir_path = training_data + "/" + dir_name
        subject_images_dir = os.listdir(subject_dir_path)

        for image_name in subject_images_dir:
            if image_name.startswith("."):
                continue

            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)

            cv2.imshow("Training on image", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)

            face, rect = face_detection(image)

            if face is not None:
                faces.append(face)
                labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey()
    cv2.destroyAllWindows()

    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(img):
    img = img.copy()

    face, rect = face_detection(img)

    label, confidence = face_recognizer.predict(face)

    label_text = subjects[label]

    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img


print("Predicting images...")

# load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")

# perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

# display both images
cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()

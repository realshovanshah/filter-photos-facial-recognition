import cv2
import face_recognition
import os

LABELLED_FACES_DIR = "known_faces"
UNLABELLED_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6

MODEL = "cnn"

labelled_faces = []
labelled_names = []

print("gathering data from labelled faces")

for foldername in os.listdir(LABELLED_FACES_DIR):
    for filename in os.listdir(f"{LABELLED_FACES_DIR}/{foldername}"):
        image = face_recognition.load_image_file(f"{LABELLED_FACES_DIR}/{foldername}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        labelled_faces.append(encoding)
        labelled_names.append(foldername)

print("gathering data from unlabelled faces")

for filename in os.listdir(UNLABELLED_FACES_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{UNLABELLED_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(labelled_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = labelled_names[results.index(True)]
            print(f"Match found: {match}")

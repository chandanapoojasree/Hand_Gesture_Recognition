import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataset_path = "leapGestRecog"

X = []
y = []

print("Loading images...")

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    for gesture in os.listdir(person_path):
        gesture_path = os.path.join(person_path, gesture)

        for image_name in os.listdir(gesture_path)[:100]:
            image_path = os.path.join(gesture_path, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, (64, 64))

            X.append(img.flatten())
            y.append(gesture)

print("Dataset Loaded!")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = SVC(kernel='linear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
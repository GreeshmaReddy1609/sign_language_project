import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model

DATA_PATH = "data"

words = sorted([w for w in os.listdir(DATA_PATH)
                if os.path.isdir(os.path.join(DATA_PATH, w))])

model = load_model("sign_model.h5")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sentence = []
prediction_buffer = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                landmarks.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])

            landmarks = np.array(landmarks).reshape(1, -1)

            prediction = model.predict(landmarks, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = prediction[predicted_class]

            prediction_buffer.append(predicted_class)

            # Keep only last 6 predictions
            if len(prediction_buffer) > 6:
                prediction_buffer = prediction_buffer[-6:]

            # Stable detection rule
            if prediction_buffer.count(prediction_buffer[-1]) > 3 and confidence > 0.6:

                word = words[predicted_class]

                if len(sentence) == 0 or word != sentence[-1]:
                    sentence.append(word)

    # Basic Grammar Fix
    sentence_text = " ".join(sentence)
    sentence_text = sentence_text.replace("I NAME", "MY NAME")
    sentence_text = sentence_text.replace("YOUR NAME", "WHAT IS YOUR NAME")
    sentence_text = sentence_text.replace("HELP YOU", "CAN YOU HELP ME")

    cv2.putText(frame, "Sentence: " + sentence_text,
                (10,40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255,0,0), 2)

    cv2.imshow("Sign Language Detection", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('c'):
        sentence = []

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





       

          

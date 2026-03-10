import cv2
import os
import numpy as np
import mediapipe as mp

# Words
words = ['HELLO','HI','YES','NO','PLEASE','THANK','WHAT','NAME','I','YOU','HELP','FINE']
samples_per_word = 50

DATA_PATH = "data"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

person_name = input("Enter person name (p1/p2/p3): ")

cap = cv2.VideoCapture(0)

for word in words:

    word_path = os.path.join(DATA_PATH, word)
    os.makedirs(word_path, exist_ok=True)

    print(f"\nCollecting data for: {word}")
    count = 0

    while count < samples_per_word:

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

                np.save(os.path.join(word_path, f"{person_name}_{count}.npy"), landmarks)
                count += 1

        cv2.putText(frame, f"{word} : {count}/{samples_per_word}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

print("\nData collection completed.")
cap.release()
cv2.destroyAllWindows()



        
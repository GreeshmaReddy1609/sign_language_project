import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

DATA_PATH = "data"

words = sorted([w for w in os.listdir(DATA_PATH)
                if os.path.isdir(os.path.join(DATA_PATH, w))])

X = []
y = []

for idx, word in enumerate(words):
    word_path = os.path.join(DATA_PATH, word)

    for file in os.listdir(word_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(word_path, file))
            X.append(data)
            y.append(idx)

if len(X) == 0:
    print("❌ No data found. Please collect data first.")
    exit()

X = np.array(X)
y = np.array(y)

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(63,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(words), activation='softmax'))

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=80,
          validation_data=(X_test, y_test))

model.save("sign_model.h5")

print("\nModel training completed and saved.")
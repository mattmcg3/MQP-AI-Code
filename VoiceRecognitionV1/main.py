# Modified from https://github.com/AssemblyAI-Examples/realtime-voice-command-recognition
import numpy as np

# Tensorflow - expects model
from tensorflow.keras import models

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer

# Modify this in the correct order (follows folders)
commands = ['close', 'down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']

# Loads model
loaded_model = models.load_model("saved_model")


# Reads command and passes to main func.
def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    # We can put Keenan's preprocessing here ^
    prediction = loaded_model(spec)
    #softmax
    #print(prediction)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    confidence = prediction[0][label_pred[0]]
    #print("Predicted label:", command)
    return command, confidence

if __name__ == "__main__":
    while True:
        command, confidence = predict_mic()
        if confidence > 1.9:
            print("Predicted label:", command)
        #     print("Predicted label: close")
        # if command == "right":
        #     print ("Predicted label: right")
        if command == "stop":
            terminate()
            break
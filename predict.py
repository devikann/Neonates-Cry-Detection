import librosa
import numpy as np
import tensorflow as tf
import sounddevice as sd

MODEL = tf.keras.models.load_model("model.h5")

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

def run_detection():
    print("Listening...")

    audio = sd.rec(SAMPLES, samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    audio = audio.flatten()

    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = mel[np.newaxis, ..., np.newaxis]

    prediction = MODEL.predict(mel)[0][0]
    print("Prediction value:", prediction)

    if prediction > 0.3:
        return "🚨 Cry Detected – Alert Doctor"
    else:
        return "No Cry Detected"

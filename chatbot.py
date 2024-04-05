import pyaudio
import numpy as np
import scipy.io.wavfile as wav
import keyboard
import wave
import librosa
import numpy as np
from joblib import load
from tkinter import messagebox
max_frames=1000

def extract_features(audio_file, max_frames=1000):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        # Calculate MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        # Pad or truncate features to have max_frames frames
        if mfcc.shape[1] < max_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_frames]
        return mfcc
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None



fs = 44100  # Sample rate


audio = pyaudio.PyAudio()
filename = "recorded.wav"
chunk = 100000
FORMAT = pyaudio.paInt16
channels = 1
sample_rate = 44100
record_seconds = 5
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=channels,
                rate=sample_rate,
                input=True,
                output=True,
                frames_per_buffer=chunk)
frames = []
print("Press space to start recording...")
keyboard.wait('space')
print("Recording started...")
for i in range(int(sample_rate / chunk * record_seconds)):
    data = stream.read(chunk)
    frames.append(data)

print("Finished recording.")
# stop and close stream
stream.stop_stream()
stream.close()

p.terminate()

wf = wave.open(filename, "wb")

wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(sample_rate)
wf.writeframes(b"".join(frames))
wf.close()

loaded_model = load("decision_tree_model.joblib")

audio_file_test =  [filename]
feature_test = [extract_features(audio_file, max_frames) for audio_file in audio_file_test]
features_flat_test = [features.flatten() for features in feature_test]
predicted_result = loaded_model.predict(features_flat_test)
messagebox.showinfo("Result", predicted_result[0])


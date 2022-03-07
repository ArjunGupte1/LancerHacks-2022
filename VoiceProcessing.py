import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from IPython.display import Audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import matplotlib.pyplot as plt
import wave
import sounddevice as sd
from scipy.io.wavfile import write
from transformers.file_utils import filename_to_url
import pyttsx3
import serial
import time
import tensorflow as tf
from PIL import Image


tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h") #NLP model - Facebook Wave2Vec
fs = 44100  # Sample rate
seconds = 3  # Duration of recording
engine = pyttsx3.init() #Create voice engine
recordingPath = "C:/Users/gupte/Documents/LancerHacks-2022/UserInputRecording.wav"
export_dir = "C:/Users/gupte/Documents/LancerHacks-2022/my_model.h5"
img_url = "./acne.jpg"
######################################################################################

def recordInput(fs, seconds):
    #print("Ready to record for " + str(seconds) + " seconds")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    write(recordingPath, fs, myrecording)  # Save as WAV file
    fileName = recordingPath
    print("Finished Recording User Input")
    return fileName #Do this just to verify where the file is

def audioProcessing(fileName):
    data = wavfile.read(fileName)
    frameRate = data[0]
    soundData = data[1]
    time = np.arange(0,len(soundData))/frameRate
    inputAudio, _ = librosa.load(fileName, sr=16000)
    inputValues = tokenizer(inputAudio, return_tensors="pt").input_values
    logits = model(inputValues).logits
    predictedIds = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predictedIds)[0].lower()
    print("User said: " + transcription)
    return transcription

def tfModel(img_url): #dummy function that will be replaced by anaiys cnn
    tf_model = tf.keras.models.load_model(export_dir)
    samp_img = Image.open(img_url)
    samp_img = samp_img.resize((400,400))
    I = np.asarray(samp_img)
    #I = np.expand_dims(I, 2)
    pred = tf_model.predict(np.array( [I,] )  ).round()
    if(np.argmax(pred[0]) == 0):
        return "Acne"
    elif(np.argmax(pred[0]) == 1):
        return "Carcinoma"
    elif(np.argmax(pred[0]) == 2):
        return "Dermatitis"
    else:
        return("Bullous Disease")
######################################################################################

#Start the program#
engine.say("Welcome to the skin disease detection program. Are you ready to begin?")
engine.runAndWait()

userInputPath = recordInput(fs, seconds)
userInputTranscription = audioProcessing(userInputPath)

while userInputTranscription != "yes":
    time.sleep(10)
    engine.say("Are you ready to begin now?")
    engine.runAndWait()
    userInputPath = recordInput(fs, seconds)
    userInputTranscription = audioProcessing(userInputPath)

engine.say("Ok great lets get started")
engine.runAndWait()

#Tell user to position themselves in front of the camera#
engine.say("Please position the affected area in front of the webcam and hold still while we take a picture")
engine.runAndWait()
time.sleep(10)

#Tell user the picture has been taken and they can move away now#
engine.say("Ok great you can move away now.")
engine.runAndWait()

#Pass image into model here#
diagnosis = tfModel(img_url)

#Tell the user the diagnosis#
engine.say("The preliminary diagnosis is {}".format(diagnosis))
engine.runAndWait()
print("The preliminary diagnosis is {}".format(diagnosis))

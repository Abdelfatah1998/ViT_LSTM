import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display

# Function to save plot as image
def save_plot_as_image(feature, output_path, file_name):
    plt.figure(figsize=(10, 4))
    sr = 16000
    display.specshow(feature, sr=sr, x_axis='time')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, file_name))
    plt.close()

# Function to extract and save features
def extract_and_save_features(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".WAV"):
            file_path = os.path.join(input_folder, file)
            y, sr = librosa.load(file_path)

            # Extract features

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            delta_mfcc = librosa.feature.delta(mfccs)
            delta2_mfcc = librosa.feature.delta(mfccs, order=2)
            zero_crossings = librosa.feature.zero_crossing_rate(y)
            mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

            # Save features as images
            save_plot_as_image(mfccs, output_folder, f"{file}_mfccs.png")
            save_plot_as_image(delta_mfcc, output_folder, f"{file}_delta_mfcc.png")
            save_plot_as_image(delta2_mfcc, output_folder, f"{file}_delta2_mfcc.png")
            save_plot_as_image(zero_crossings, output_folder, f"{file}_zero_crossings.png")
            save_plot_as_image(mel_spectrogram, output_folder, f"{file}_mel_spectrogram.png")
            save_plot_as_image(spectral_rolloff, output_folder, f"{file}_spectral_rolloff.png")
            save_plot_as_image(rms, output_folder, f"{file}_rms.png")
            save_plot_as_image(chroma, output_folder, f"{file}_chroma.png")
            save_plot_as_image(spectral_bandwidth, output_folder, f"{file}_spectral_bandwidth.png")
            save_plot_as_image(spectral_centroid, output_folder, f"{file}_spectral_centroid.png")
            save_plot_as_image(tonnetz, output_folder, f"{file}_tonnetz.png")

# Paths to input and output folders
input_folder = r'C:\Users\fta71\PycharmProjects\pythonProject2\ViT_LSTM\Input'
output_folder = r'C:\Users\fta71\PycharmProjects\pythonProject2\ViT_LSTM\Output'

# Extract and save features
extract_and_save_features(input_folder, output_folder)

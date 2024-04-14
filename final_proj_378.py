from scipy.io import wavfile as wv
import scipy.signal as ss
from scipy.stats import entropy
import numpy as np
import os
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import sklearn.naive_bayes as sknb
td = pd.read_csv('train.csv')
y = td['Genre'].values

directory_test = "test"

audio_data_list_test = []
sampling_rate_list_test = []

# Iterate over training data
for filename in os.listdir(directory_test):
    # Loads all the .wav files in the traning set
    if filename.endswith(".wav"):
        file_path = os.path.join(directory_test, filename)
        audio_data, sampling_rate = librosa.load(file_path, sr=None)
        
        # Append the audio data and sampling rate to the lists
        audio_data_list_test.append(audio_data)
        sampling_rate_list_test.append(sampling_rate)

# Now we can use the audio data and sampling rate data

print("Test data loaded!")

directory = "train"

audio_data_list = []
sampling_rate_list = []

# Iterate over training data
for filename in os.listdir(directory):
    # Loads all the .wav files in the traning set
    if filename.endswith(".wav"):
        file_path = os.path.join(directory, filename)
        audio_data, sampling_rate = librosa.load(file_path, sr=None)
        
        # Append the audio data and sampling rate to the lists
        audio_data_list.append(audio_data)
        sampling_rate_list.append(sampling_rate)

# Now we can use the audio data and sampling rate data

print("Train data loaded!")

# Gathers the frequency information of a song
fft_list = []
for song in audio_data_list:
    fft_list.append(np.fft.fft(song))
    
fft_list_test = []
for song in audio_data_list_test:
    fft_list_test.append(np.fft.fft(song))

print("Freq data done!")

# Computes energy of each song
energy_list = []
for song in audio_data_list:
    energy_list.append(np.sum(np.square(song)))
    
energy_list_test = []
for song in audio_data_list_test:
    energy_list_test.append(np.sum(np.square(song)))

print("Energy data done!")

# Finds the variance and mean of each song
variance_list = []
expectation_list = []
for song in audio_data_list:
    variance_list.append(np.var(song))
    expectation_list.append(np.mean(song))
    
variance_list_test = []
expectation_list_test = []
for song in audio_data_list_test:
    variance_list_test.append(np.var(song))
    expectation_list_test.append(np.mean(song))

print("Var/mean data done!")

tempo_list = []
for i in range(0,len(audio_data_list)):
    tempo, beats = librosa.beat.beat_track(y=audio_data_list[i], sr=sampling_rate_list[i])
    tempo_list.append(tempo)

tempo_list_test = []
for i in range(0,len(audio_data_list_test)):
    tempo, beats = librosa.beat.beat_track(y=audio_data_list_test[i], sr=sampling_rate_list_test[i])
    tempo_list_test.append(tempo)

print("Tempo data done!")

zero_cross_rate_list = []
for song in audio_data_list:
    zero_cross_rate_list.append(np.median(librosa.feature.zero_crossing_rate(song)))

zero_cross_rate_list_test = []
for song in audio_data_list_test:
    zero_cross_rate_list_test.append(np.median(librosa.feature.zero_crossing_rate(song)))

print("ZCR data done!")

fft_entrp_list = []
for fft in fft_list:
    fft_entrp_list.append(entropy(np.absolute(fft)))

fft_entrp_list_test = []
for fft in fft_list_test:
    fft_entrp_list_test.append(entropy(np.absolute(fft)))

print("FFT entr data done!")

def conv_compare(song, data):
    """
    Sees how similar the fft of the songs are, higher number means more overlap
    song is one song and data is all the songs to compare song to
    can be used for time or for freq domain info
    """
    flipped_song = np.flip(song)
    convolution_sim_list = []
    for audio in data:
        sum = 0
        for i in range(0,len(audio)):
            sum += audio[i]*flipped_song[i]
        convolution_sim_list.append(sum)

# Step 1: Prepare the Feature Matrix
feature_matrix = np.column_stack((energy_list, variance_list, expectation_list, tempo_list, zero_cross_rate_list, fft_entrp_list))
feature_matrix_test = np.column_stack((energy_list_test, variance_list_test, expectation_list_test, tempo_list_test, zero_cross_rate_list_test, fft_entrp_list_test))

# Step 2: Normalize the Feature Matrix
scaler = StandardScaler()
normalized_features = scaler.fit_transform(feature_matrix)
normalized_features_test = scaler.fit_transform(feature_matrix_test)

# Step 3: Apply ICA
ica = FastICA(n_components=10, random_state=42)
independent_components = ica.fit_transform(normalized_features)
independent_components_test = ica.fit_transform(normalized_features_test)

# Step 4: Clustering and Classifying
clf = SVC(kernel='linear')
clf.fit(independent_components, y)

# Map cluster labels to genres
predicted_genres = clf.predict(independent_components_test)
nb_model = sknb.GaussianNB()
nb_model.fit(normalized_features, y)
predicted_genres = nb_model.predict(normalized_features_test)
predictions = []
for filename, genre in zip(os.listdir(directory_test), predicted_genres):
    predictions.append((filename,genre))

# Sort the list nurmerically since the dict is random
sorted_data = sorted(predictions, key=lambda x: x[0][5:8])

# Puts it into a .csv file (finally!!!!)
df = pd.DataFrame(sorted_data, columns=['ID', 'genre'])
df.to_csv('test.csv', index=False)
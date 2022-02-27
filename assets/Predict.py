import librosa, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

final = pd.read_pickle("extracted_df.pkl")
y = np.array(final["class"].tolist())
le = LabelEncoder()
le.fit_transform(y)
Model1_ANN = load_model("Model1.h5")
Model2_CNN1D = load_model("Model2.h5")
Model3_CNN2D = load_model("Model3.h5")


def extract_feature(audio_path):
    audio_data, sample_rate = librosa.load(audio_path, res_type="kaiser_fast")
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
    feature_scaled = np.mean(feature.T, axis=0)
    return np.array([feature_scaled])


def ANN_print_prediction(audio_path):
    prediction_feature = extract_feature(audio_path)
    predicted_vector = np.argmax(Model1_ANN.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]


def CNN1D_print_prediction(audio_path):
    tmp = extract_feature(audio_path)
    prediction_feature = np.expand_dims(tmp, axis=2)
    predicted_vector = np.argmax(Model2_CNN1D.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]


def CNN2D_print_prediction(audio_path):
    tmp2 = extract_feature(audio_path)
    prediction_feature = tmp2.reshape(tmp2.shape[0], 16, 8, 1)
    predicted_vector = np.argmax(Model3_CNN2D.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]


audio_dataset_path = "D:/UrbanSound8K/"
path = audio_dataset_path + "fold8/103076-3-0-0.wav"
print("\nANN Model Output --> ", ANN_print_prediction(path))
print("\nCNN1D Model Output --> ", CNN1D_print_prediction(path))
print("\nCNN2D Model Output --> ", CNN2D_print_prediction(path))

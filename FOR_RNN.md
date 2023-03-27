# Artificial-neural-networks

RNN ile Havayolu Yolcu Tahmini
Bu projede, havayolu yolcu sayısını tahmin etmek için Tekrarlayan Sinir Ağı (RNN) modeli kuracağız. Kullanacağımız veri seti, Kaggle'dan Uluslararası Havayolu Yolcuları veri setidir.

veri kümesi
Uluslararası Havayolu Yolcuları veri seti, Ocak 1949'dan Aralık 1960'a kadar aylık toplam uluslararası havayolu yolcularını içerir.

Gereksinimler
numpy
pandas
datetime
tensorflow
matplotlib
scikit-learn


Veri Ön İşleme
Gerekli kitaplıkları içe aktardıktan sonra, veri setini bir pandas DataFrame içine yüklüyoruz ve sütunları yeniden adlandırıyoruz. Ardından, eksik verileri içeren son satırı bırakırız. Boş değerleri kontrol ediyoruz ve "Ay" sütununu tarih saat biçimine dönüştürüyoruz. "Month" sütununu DataFrame'in indeksi olarak ayarlıyoruz ve "Month" sütununu bırakıyoruz. Son olarak verileri çiziyoruz.


import numpy as np 
import pandas as pd 
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , SimpleRNN,Dropout
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

df=pd.read_csv("/international-airline-passengers.csv")
df.columns=["Month","Passengers"]
df=df[:-1]
df["Month"]=pd.to_datetime(df["Month"])
df.index=df["Month"]
df.drop("Month",axis=1,inplace=True)

df.plot(figsize=(14,8),title='Monthly airline passengers');


Daha sonra verileri bir numpy dizisine dönüştürür ve yeniden şekillendiririz.

data=df["Passengers"].values.astype('float32')
data=data.reshape(-1,1)


Fonksiyonu kullanarak verileri eğitim ve test kümelerine ayırdık split_data().

def split_data(dataframe,test_size):
  position=int(round(len(dataframe)*(1-test_size)))
  train=dataframe[:position]
  test=dataframe[position:]
  return train,test,position

train,test,position=split_data(data,0.33)


Fonksiyonu kullanarak verileri ölçeklendiriyoruz MinMaxScaler().

scaler=MinMaxScaler(feature_range=(0,1))
train=scaler.fit_transform(train)
test=scaler.transform(test)


Giriş-çıkış çiftlerimizi oluşturmak için bir fonksiyon oluşturuyoruz.

def create_dataset(data,k):
    X, y = [], []
    for i in range(len(data)-k-1):
        X.append(data[i:(i+k), 0])
        y.append(data[(i+k), 0])
    return np.array(X), np.array(y)

k = 1
X_train, y_train = create_dataset(train, k)
X_test, y_test = create_dataset(test, k)

Verilerimizi modelimizin giriş şekline uyacak şekilde yeniden şekillendiriyoruz.



X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
modeli

Basit bir RNN katmanı ve ardından yoğun bir çıktı katmanı ile sıralı bir model oluşturuyoruz. Fazla uydurmayı önlemek için RNN katmanımıza bırakma düzenlileştirmesi ekliyoruz. Modeli, ortalama kare hata kaybı işlevini ve Adam iyileştiriciyi kullanarak derliyoruz.



model = Sequential()
model.add(SimpleRNN(units=50, input_shape=(X_train.shape[1], 1), dropout=0.2))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

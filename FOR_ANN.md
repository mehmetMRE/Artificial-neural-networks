# Artificial-neural-networks
Mnist
Bu proje, TensorFlow'da bir yapay sinir ağı (ANN) kullanarak MNIST veri seti üzerinde sınıflandırma yapmayı amaçlar.



Veri Seti
Bu proje, Keras'taki MNIST veri setini kullanmaktadır. MNIST, 28x28 boyutunda siyah beyaz el yazısı rakamlardan oluşan bir veri kümesidir. Eğitim veri seti 60.000 görüntü içerirken, test veri seti 10.000 görüntü içermektedir.



Kurulum
Bu projeyi çalıştırmak için şunları yüklemeniz gerekmektedir:

TensorFlow
NumPy
Matplotlib


Ayrıca, MNIST veri setine doğrudan erişmek için Keras'a da ihtiyacınız olacaktır. Bu, TensorFlow'un yüklü olduğundan emin olduğunuzda şu şekilde yapılabilir:
from tensorflow.keras.datasets import mnist
(x_train,y_train), (x_test,y_test) = mnist.load_data()


Model
Yapay sinir ağı, bir giriş katmanı, bir gizli katman ve bir çıkış katmanı içerir. Giriş katmanı, 28x28 boyutundaki görüntülerin piksel değerlerini alır. Gizli katmanda 128 nöron bulunur ve ReLU aktivasyon fonksiyonu kullanılır. Çıkış katmanı, 10 nörona sahiptir ve Softmax aktivasyon fonksiyonu kullanılır.

Model, 'categorical_crossentropy' kayıp fonksiyonu, 'adam' optimizer ve 'accuracy', 'precision' ve 'recall' metriklerini kullanarak derlenir.


Eğitim
Model, 5 epoch boyunca 128 batch boyutuyla eğitilir. Ayrıca, doğrulama veri kümesi kullanılarak modelin doğruluğu izlenir.
model.fit(x_train,y_train,epochs=5,batch_size=128,validation_data=(x_test,y_test))



Sonuçlar
Bu projede kullanılan yapay sinir ağı, MNIST veri setinde %98.8 doğruluk oranı elde etti.

Daha fazla bilgi
Daha fazla bilgi için, bu projeyle birlikte verilen kodları ve belgeleri inceleyebilirsiniz.

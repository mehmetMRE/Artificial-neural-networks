# Artificial-neural-networks
CNN ile Görüntü Sınıflandırma
Bu, TensorFlow ve Keras ile Evrişimli Sinir Ağları (CNN) kullanan bir görüntü sınıflandırma uygulamasıdır. Modelin amacı, farklı çöp türlerinin görüntülerini altı kategoride sınıflandırmaktır: karton, cam, metal, kağıt, plastik ve çöp.

Önkoşullar

TensorFlow
Keras
OpenCV
NumPy
Pandas
Seaborn
Matplotlib
Scikit-learn


Kurulum
Gerekli kütüphaneleri pip kullanarak kurabilirsiniz:

pip install tensorflow keras opencv-python numpy pandas seaborn matplotlib scikit-learn



veri kümesi
Bu uygulama için kullanılan veri kümesi şu bağlantıda bulunabilir: https://www.kaggle.com/asdasdasasdas/garbage-classification

Veri kümesi, her biri belirli bir çöp türünün görüntülerini içeren 6 klasörden oluşur. Modeli eğitmeden önce, görüntüler 224 x 224 piksel olarak yeniden boyutlandırılır.

Kullanım
Aşağıdakileri kullanarak bu depoyu yerel makinenize kopyalayın:


git clone https://github.com/<username>/<repository>.git


model
Model, Evrişimli ve Yoğun katmanların birkaç katmanından oluşur. Kayıp fonksiyonu olarak Adam iyileştiriciyi ve kategorik çapraz entropiyi kullanır. Modelin performansı, doğruluk, kesinlik ve geri çağırma metrikleri kullanılarak izlenir.

Sonuçlar
Model, 100 çağdan sonra %91,9'luk bir eğitim doğruluğu ve %87,7'lik bir doğrulama doğruluğu elde etti. Eğitim ve doğrulama kaybı değerleri, aşağıdaki şekilde dönem sayısına göre çizilir:


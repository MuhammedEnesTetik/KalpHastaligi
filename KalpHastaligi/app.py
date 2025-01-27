import pickle

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import LabelEncoder

##### Veri Setinin Okunması
risk = pd.read_csv('heart.csv')

##### Veri Setinin Sütunlarının Yeniden İsimlendirilmesi - Veri Önişleme Başlangıç
risk = risk.rename(columns = {
    "Age" : "Yaş" ,
    "Sex" : "Cinsiyet" , 
    "ChestPainType" : "Göğüs_Ağrısı_Tipi" ,
    "RestingBP" : "Kan_Basıncı" ,
    "Cholesterol" : "Kolesterol" ,
    "FastingBS" : "Kan_Şekeri" ,
    "RestingECG" : "İstirahat_EDS" ,
    "MaxHR" : "Maks_Kalp_Atış" ,
    "ExerciseAngina" : "Egzersize_Bağlı_Anjina" ,
    "Oldpeak" : "Egzersiz_ST_Depresyonu" ,
    "ST_Slope" : "ST_Segment_Eğimi" ,
    "HeartDisease" : "Durum" 
    })


categorical_features = risk.select_dtypes(include=['object']).columns
numerical_features = risk.select_dtypes(include=['int64', 'float64']).columns

label_encoder = LabelEncoder()
for feature in categorical_features:
    risk[feature] = label_encoder.fit_transform(risk[feature])

### Makine Öğrenmesi Başlangıç

##### Veri Setinin Bağımlı ve Bağımsız Değişken Ayrımı
y=risk['Durum']
x=risk.drop(columns=['Durum'],axis=1)

##### Veri Setinin Train ve Test ayrılması ve Oran Belirleme
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=65)

## En İyi Parametreler:{'learning_rate': 0.05, 'max_depth': 3, 'min_samples_split': 5, 'n_estimators': 100}
##### Makine Öğrenmesi Algoritması / Model
gbm = GradientBoostingClassifier(learning_rate = 0.05,
                                 max_depth = 3,
                                 min_samples_split = 5,
                                 n_estimators = 100)

##### Öğrenme
model = gbm.fit(x_train,y_train)


# Modeli kaydet
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

from flask import Flask, render_template, request, redirect, url_for , jsonify
import pyodbc
import hashlib

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Veritabanı bağlantı dizesini belirleyin
server = r'ENES\SQLEXPRESS'
database = 'kalp_hastaligi'
conn_str = f'DRIVER=ODBC Driver 17 for SQL Server;SERVER={server};DATABASE={database};Trusted_Connection=yes;'

#----------------------------------------------------------------------------------------------------------------------------------------#

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/kayit', methods=['GET', 'POST'])
def kayit():
    if request.method == 'POST':
        kullanici_adi = request.form['kullanici_adi']
        eposta = request.form['eposta']
        parola = request.form['parola']
        
        try:
            # Parolayı hashle
            hash_parola = hashlib.sha256(parola.encode()).digest()

            # Veritabanına bağlanın
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()

            # Yeni kullanıcıyı veritabanına ekle
            cursor.execute("INSERT INTO Kullanicilar (KullaniciAdi, Eposta, ParolaHash) VALUES (?, ?, ?)", (kullanici_adi , eposta, hash_parola))
            conn.commit()

            return redirect(url_for('giris'))

        except Exception as e:
            return render_template('index.html', error=str(e))

        finally:
            # Bağlantıyı kapatın
            if 'conn' in locals():
                conn.close()
    
    return render_template('index.html')

@app.route('/giris', methods=['GET', 'POST'])
def giris():
    if request.method == 'POST':
        eposta = request.form['eposta']
        parola = request.form['parola']
        
        try:
            # Parolayı hashle
            hash_parola = hashlib.sha256(parola.encode()).digest()

            # Veritabanına bağlanın
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()

            # E-postaya karşılık gelen parola hash'ini alın
            cursor.execute("SELECT ParolaHash FROM Kullanicilar WHERE Eposta = ?", (eposta,))
            stored_hash = cursor.fetchone()

            if stored_hash:
                # Girilen parolanın hash değeri ile veritabanındaki hash değerini karşılaştırın
                if stored_hash[0] == hash_parola:
                    return render_template('veri_ay.html')
                else:
                    return "E-Posta veya parola yanlış!"
            else:
                return "Kullanıcı bulunamadı."

        except Exception as e:
            return render_template('index.html', error=str(e))

        finally:
            # Bağlantıyı kapatın
            if 'conn' in locals():
                conn.close()
    
    return render_template('index.html')


import pickle       

# Modeli yükle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # JSON formatında gelen veriyi al
    features = [data['yaş'], data['cinsiyet'], data['gat'], data['kb'], data['kolesterol'], data['kş'], data['ieds'], data['mka'], data['eba'], data['estd'], data['eğim']]
    prediction = model.predict([features])
    probability = model.predict_proba([features])[0][1]
    return jsonify({'prediction': int(prediction), 'probability': float(probability)})


if __name__ == '__main__':
    app.run(debug=True)

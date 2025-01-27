import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

from sklearn.preprocessing import LabelEncoder
##### Veri Setinin Okunması
risk = pd.read_csv('heart.csv')
##### Veri Setinin İlk 5 Satırı
#risk.head(5)
#risk.describe().T
#risk["HeartDisease"].value_counts()
#risk["HeartDisease"].value_counts().plot.barh();
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
#risk.head(50)
categorical_features = risk.select_dtypes(include=['object']).columns
numerical_features = risk.select_dtypes(include=['int64', 'float64']).columns
print("\nCategorical Features:", categorical_features)
print("\nNumerical Features:", numerical_features)
label_encoder = LabelEncoder()
for feature in categorical_features:
    risk[feature] = label_encoder.fit_transform(risk[feature])
#risk.head(15)
##### Veri Önişleme Devamı
#risk.info()
#risk.isnull().sum()
### Makine Öğrenmesi Başlangıç
##### Veri Setinin Bağımlı ve Bağımsız Değişken Ayrımı
y=risk['Durum']
x=risk.drop(columns=['Durum'],axis=1)
# y = np.ravel(y)
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

'''
#### Bağımsız değişkenlerin model tahmini üzerindeki etkisi / önemi
#risk.head(1)
#özellik_önemi = gbm.feature_importances_


#for i, importance in enumerate(özellik_önemi):
 #   print(f"Özellik {i+1}: Önem = {importance}")
### Model Metrikleri
#y_pred = model.predict(x_test)
##### Doğruluk Skoru
#accuracy_score(y_test, y_pred)
#model.score(x_test,y_test)
##### Karışıklık Matrisi
#confusion_matrix(y_test, y_pred)
#print(classification_report(y_test, y_pred))
#accuracy_score(y_test, y_pred)
#### Kesinlik
#precision_score(y_test, y_pred)
#### Duyarlılık
#recall_score(y_test, y_pred)
#### F1 Score
#f1_score(y_test, y_pred)


#### ROC
print(roc_curve(y_test, y_pred))
print(roc_curve(y_test, y_pred))
model.predict_proba(x_test) [:,1] [0:5]
gbm_roc_auc = roc_auc_score(y_test, y_pred)

fpr, tpr, tresholds = roc_curve(y_test, model.predict_proba(x_test) [:,1])

plt.figure()
plt.plot(fpr, tpr, label = 'AUC (area = %0.2f)' % gbm_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()

'''

# Modeli kaydet
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

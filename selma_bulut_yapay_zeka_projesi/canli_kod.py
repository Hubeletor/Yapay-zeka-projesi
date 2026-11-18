# -*- coding: utf-8 -*-
"""
YAPAY ZEKA DERSİ - GRUP 5 PROJESİ
Konu     : Gözetimli Öğrenme - Karar Ağaçları & Ensemble Yöntemler
Algoritma: Random Forest & XGBoost Sınıflandırıcı
Veri seti: Wisconsin Meme Kanseri Teşhis Veri Seti (UCI / scikit-learn)
           569 hasta, 30 sayısal öznitelik, hedef: habis (kötü huylu) mu?

Sunumda bölüm bölüm (# %% hücreleri) çalıştırın.
VS Code / PyCharm / Spyder'da "# %%" satırları hücre olarak çalışır,
ya da aynı içeriğin olduğu canli_kod.ipynb dosyasını Jupyter'de açın.
"""

# %% 1) KÜTÜPHANELER VE AYARLAR
# Bu bölümde ne yapıyoruz?
# - Projede kullanacağımız kütüphaneleri yüklüyoruz: pandas / numpy veriyle çalışmak, matplotlib / seaborn grafik çizmek, scikit-learn modeller ve metrikler, xgboost XGBoost modeli için.
# - Grafiklerin kaydedileceği grafikler/ ve verinin kaydedileceği veri/ klasörlerini oluşturuyoruz.
# - RASTGELE = 42 ile rastgeleliği sabitliyoruz. Böylece kod her çalıştığında aynı sonuçları verir.
# - kaydet() fonksiyonu her grafiği hem ekranda gösterir hem de dosyaya kaydeder.
#
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, RocCurveDisplay)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110

KLASOR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
GRAFIK = os.path.join(KLASOR, "grafikler")
VERI = os.path.join(KLASOR, "veri")
os.makedirs(GRAFIK, exist_ok=True)
os.makedirs(VERI, exist_ok=True)
RASTGELE = 42


def kaydet(ad):
    """Grafiği grafikler/ klasörüne kaydeder ve gösterir."""
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFIK, ad), bbox_inches="tight")
    plt.show()


# %% 2) VERİ SETİNİ ÇEKME (YÜKLEME)
# Bu bölümde ne yapıyoruz?
# - Wisconsin Meme Kanseri veri setini scikit-learn'ün içinden yüklüyoruz. İnternet gerekmiyor.
# - Veri, Wisconsin Üniversitesi'nde 1995'te hazırlanmış ve UCI Machine Learning Repository'de yayınlanmış: 569 hasta, 30 sayısal öznitelik.
# - Orijinal veride 0 = habis, 1 = iyi huylu. Biz bunu ters çevirip 1 = Habis yapıyoruz, çünkü asıl yakalamak istediğimiz sınıf habis tümörler.
# - Veriyi veri/meme_kanseri.csv olarak da kaydediyoruz.
#
# Veri seti scikit-learn ile birlikte gelir; internet gerekmez.
# Orijinal kodlamada 0 = habis, 1 = iyi huylu. Habis sınıfı "pozitif" (1) olsun diye ters çeviriyoruz:
ham = load_breast_cancer(as_frame=True)
df = ham.frame.copy()
df["hedef"] = 1 - df["target"]          # 1 = Habis (kötü huylu), 0 = İyi huylu
df = df.drop(columns="target")
df.to_csv(os.path.join(VERI, "meme_kanseri.csv"), index=False)   # CSV olarak da saklıyoruz

print("Veri boyutu:", df.shape)
print("Öznitelik sayısı:", df.shape[1] - 1)
print("Eksik değer sayısı:", int(df.isna().sum().sum()))
df.head()

# %% 3) KEŞİFSEL VERİ ANALİZİ (EDA)
# Bu bölümde ne yapıyoruz? (Keşifsel Veri Analizi)
# - Modeli kurmadan önce veriyi tanıyoruz.
# - Kaç hastanın habis, kaçının iyi huylu olduğunu sayıyoruz: 357 iyi huylu, 212 habis.
# - İlk 10 özniteliğin ortalama, standart sapma, minimum ve maksimum gibi istatistiklerini yazdırıyoruz.
# - Sınıf dağılımını çubuk grafikle gösteriyoruz. Sınıflar tam dengeli olmadığı için sadece Accuracy'ye bakmak yetmez.
#
print(df["hedef"].value_counts().rename({0: "İyi huylu", 1: "Habis"}))
print()
print(df.describe().T.head(10).round(2))

# 3.1 Sınıf dağılımı
plt.figure(figsize=(5, 4))
ax = sns.countplot(x=df["hedef"].map({0: "İyi huylu", 1: "Habis"}))
ax.set_title("Sınıf Dağılımı")
ax.set_xlabel("")
ax.set_ylabel("Hasta sayısı")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
kaydet("01_sinif_dagilimi.png")

# %% 3.2 Korelasyon ısı haritası (hedefle en ilişkili 10 öznitelik)
# Bu bölümde ne yapıyoruz?
# - Teşhisle (hedef) en güçlü ilişkisi olan 10 özniteliği seçiyoruz.
# - Bu özniteliklerin hem birbirleriyle hem de hedefle olan korelasyonunu renkli bir tabloda (ısı haritası) gösteriyoruz.
# - Kırmızıya yakın renkler güçlü pozitif ilişki demek. Yarıçap, çevre ve alan gibi boyut ölçüleri habis olmayla güçlü şekilde ilişkili.
#
en_iliskili = df.corr()["hedef"].abs().sort_values(ascending=False).index[1:11]
plt.figure(figsize=(8, 6))
sns.heatmap(df[list(en_iliskili) + ["hedef"]].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, annot_kws={"size": 7})
plt.title("Korelasyon Matrisi (hedefle en ilişkili 10 öznitelik)")
kaydet("02_korelasyon.png")

# %% 3.3 Seçili özniteliklerin sınıfa göre dağılımı
# Bu bölümde ne yapıyoruz?
# - 4 özniteliğin habis ve iyi huylu hastalardaki dağılımını histogramla üst üste çiziyoruz.
# - Habis tümörlerde bu değerler belirgin şekilde daha büyük. İki dağılım az örtüşüyor, yani model iki sınıfı ayırabilir.
#
secili = ["worst perimeter", "worst concave points", "mean radius", "mean texture"]
fig, eksenler = plt.subplots(2, 2, figsize=(9, 6))
for ax, sutun in zip(eksenler.ravel(), secili):
    sns.histplot(data=df, x=sutun, hue=df["hedef"].map({0: "İyi huylu", 1: "Habis"}),
                 kde=True, ax=ax, element="step")
    ax.set_title(sutun)
    ax.legend_.set_title("")
fig.suptitle("Özniteliklerin Sınıfa Göre Dağılımı", y=1.01)
kaydet("03_oznitelik_dagilimlari.png")

# %% 4) EĞİTİM / TEST AYIRMA
# Bu bölümde ne yapıyoruz?
# - Veriyi %80 eğitim (455 hasta) ve %20 test (114 hasta) olarak ikiye bölüyoruz.
# - Model test verisini eğitimde hiç görmüyor. Gerçek başarıyı bu görülmemiş veride ölçüyoruz.
# - stratify=y sayesinde iki parçada da habis hasta oranı aynı kalıyor.
# - Ölçekleme (StandardScaler) yapmıyoruz. Ağaç modelleri "bu değer şu eşikten büyük mü?" diye karar verdiği için sayıların birimi sonucu etkilemiyor.
#
X = df.drop(columns="hedef")
y = df["hedef"]
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RASTGELE)
print("Eğitim:", X_egitim.shape, "| Test:", X_test.shape)
# Not: Ağaç tabanlı modeller ölçeklemeye (StandardScaler) ihtiyaç duymaz.

# %% 5) MODELLERİN EĞİTİMİ
# Bu bölümde ne yapıyoruz? Üç model kurup eğitiyoruz:
# - Karar Ağacı: Tek bir ağaç. Diğer modelleri kıyaslamak için temel model.
# - Random Forest: 200 ağaç. Her ağaç verinin rastgele bir örneğiyle eğitilir, sonuç oy çokluğuyla belirlenir (bagging).
# - XGBoost: 300 ağaç. Ağaçlar sırayla eğitilir, her yeni ağaç öncekilerin hatasını düzeltir (boosting). Ayarlar: öğrenme hızı 0.05, derinlik 4, subsample 0.8.
#
# Her model için test hastalarının tahminini ve habis olma olasılığını saklıyoruz.
#
modeller = {
    "Karar Ağacı": DecisionTreeClassifier(random_state=RASTGELE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RASTGELE, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=RASTGELE),
}

tahminler, olasiliklar = {}, {}
for ad, model in modeller.items():
    model.fit(X_egitim, y_egitim)
    tahminler[ad] = model.predict(X_test)
    olasiliklar[ad] = model.predict_proba(X_test)[:, 1]
    print(f"{ad:15s} eğitildi.")

# %% 6) BAŞARI METRİKLERİ
# Bu bölümde ne yapıyoruz? Üç modelin test setindeki başarısını 5 metrikle ölçüyoruz:
# - Accuracy: Tüm hastaların yüzde kaçı doğru sınıflandı?
# - Precision: "Habis" dediklerimizin yüzde kaçı gerçekten habis?
# - Recall: Gerçek habislerin yüzde kaçını yakaladık? Bu problemde en önemli metrik budur.
# - F1: Precision ve Recall'un dengeli ortalaması.
# - ROC-AUC: Model habis ile iyi huyluyu ne kadar iyi ayırıyor? 1'e ne kadar yakınsa o kadar iyi.
#
# Sonuçları tabloya yazıyor, veri/model_sonuclari.csv olarak kaydediyor ve çubuk grafikle karşılaştırıyoruz.
#
satirlar = []
for ad in modeller:
    satirlar.append({
        "Model": ad,
        "Accuracy": accuracy_score(y_test, tahminler[ad]),
        "Precision": precision_score(y_test, tahminler[ad]),
        "Recall": recall_score(y_test, tahminler[ad]),
        "F1": f1_score(y_test, tahminler[ad]),
        "ROC-AUC": roc_auc_score(y_test, olasiliklar[ad]),
    })
sonuc = pd.DataFrame(satirlar).set_index("Model").round(4)
sonuc.to_csv(os.path.join(VERI, "model_sonuclari.csv"))
print(sonuc)

sonuc.plot(kind="bar", figsize=(9, 4.5), ylim=(0.85, 1.01))
plt.title("Model Karşılaştırması (Test Seti)")
plt.xticks(rotation=0)
plt.legend(loc="lower right", ncol=5, fontsize=8)
kaydet("04_model_karsilastirma.png")

# %% 7) KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX)
# Bu bölümde ne yapıyoruz?
# - Her model için 2×2'lik karmaşıklık matrisini çiziyoruz. Satırlar gerçek durumu, sütunlar modelin tahminini gösteriyor.
# - Sol alt hücre en kritik hata: habis olduğu hâlde "iyi huylu" denen, yani kaçırılan hastalar (yanlış negatif).
# - Ardından Random Forest ve XGBoost için sınıf bazında ayrıntılı bir rapor yazdırıyoruz.
#
fig, eksenler = plt.subplots(1, 3, figsize=(13, 4))
for ax, ad in zip(eksenler, modeller):
    ConfusionMatrixDisplay(confusion_matrix(y_test, tahminler[ad]),
                           display_labels=["İyi huylu", "Habis"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(ad)
kaydet("05_karmasiklik_matrisleri.png")

print("Random Forest - Sınıflandırma Raporu")
print(classification_report(y_test, tahminler["Random Forest"], target_names=["İyi huylu", "Habis"]))
print("XGBoost - Sınıflandırma Raporu")
print(classification_report(y_test, tahminler["XGBoost"], target_names=["İyi huylu", "Habis"]))

# %% 8) ROC EĞRİSİ
# Bu bölümde ne yapıyoruz?
# - Üç modelin ROC eğrisini ve rastgele tahmini gösteren kesikli çizgiyi aynı grafikte çiziyoruz.
# - Eğri sol üst köşeye ne kadar yakınsa model o kadar iyi.
# - Eğrinin altında kalan alan ROC-AUC değeridir.
#
fig, ax = plt.subplots(figsize=(6, 5))
for ad, model in modeller.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=ad)
ax.plot([0, 1], [0, 1], "k--", label="Rastgele tahmin")
ax.set_title("ROC Eğrileri")
ax.legend(loc="lower right")
kaydet("06_roc_egrisi.png")

# %% 9) ÇAPRAZ DOĞRULAMA (5 KATLI)
# Bu bölümde ne yapıyoruz?
# - Tek bir eğitim/test bölünmesine güvenmemek için veriyi 5 parçaya bölüyoruz.
# - Her model 5 kez eğitiliyor ve her seferinde farklı bir parçada test ediliyor.
# - Ortalama doğruluğu ve standart sapmayı yazdırıyoruz. Kutu grafiği modellerin ne kadar kararlı olduğunu gösteriyor.
# - Veri küçük (569 hasta) olduğu için bu adım sonuçlara güvenmemizi sağlıyor.
#
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RASTGELE)
cv_sonuc = {}
for ad, model in modeller.items():
    skor = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    cv_sonuc[ad] = skor
    print(f"{ad:15s} ortalama: {skor.mean():.4f}  (± {skor.std():.4f})")

plt.figure(figsize=(6, 4))
sns.boxplot(data=pd.DataFrame(cv_sonuc))
plt.title("5 Katlı Çapraz Doğrulama Doğruluğu")
plt.ylabel("Accuracy")
kaydet("07_capraz_dogrulama.png")

# %% 10) ÖZNİTELİK ÖNEMLERİ
# Bu bölümde ne yapıyoruz?
# - Random Forest ve XGBoost'un karar verirken en çok dayandığı 10 özniteliği gösteriyoruz.
# - Böylece model bir kara kutu olmaktan çıkıyor. En belirleyici öznitelikler tümörün en kötü bölgesindeki boyut ve girinti ölçümleri (worst perimeter, worst area, worst concave points).
#
onem = pd.DataFrame({
    "Random Forest": modeller["Random Forest"].feature_importances_,
    "XGBoost": modeller["XGBoost"].feature_importances_,
}, index=X.columns)

fig, eksenler = plt.subplots(1, 2, figsize=(13, 5))
for ax, ad in zip(eksenler, onem.columns):
    onem[ad].sort_values().tail(10).plot(kind="barh", ax=ax, color="#66a61e" if ad == "XGBoost" else "#1b9e77")
    ax.set_title(f"{ad} - En Önemli 10 Öznitelik")
kaydet("08_oznitelik_onemleri.png")

# %% 11) HİPERPARAMETRE ETKİSİ: AĞAÇ SAYISI
# Bu bölümde ne yapıyoruz?
# - İki modeli 10, 25, 50, 100, 200 ve 300 ağaçla ayrı ayrı eğitip test doğruluğunu çiziyoruz.
# - Amaç bir hiperparametrenin (n_estimators, yani ağaç sayısı) başarıya etkisini görmek.
# - Belli bir sayıdan sonra ağaç eklemek başarıyı artırmıyor, sadece modeli yavaşlatıyor.
#
agac_sayilari = [10, 25, 50, 100, 200, 300]
rf_skor, xgb_skor = [], []
for n in agac_sayilari:
    rf = RandomForestClassifier(n_estimators=n, random_state=RASTGELE, n_jobs=-1).fit(X_egitim, y_egitim)
    xg = XGBClassifier(n_estimators=n, learning_rate=0.05, max_depth=4, eval_metric="logloss",
                       random_state=RASTGELE).fit(X_egitim, y_egitim)
    rf_skor.append(accuracy_score(y_test, rf.predict(X_test)))
    xgb_skor.append(accuracy_score(y_test, xg.predict(X_test)))

plt.figure(figsize=(7, 4))
plt.plot(agac_sayilari, rf_skor, "o-", label="Random Forest")
plt.plot(agac_sayilari, xgb_skor, "s-", label="XGBoost")
plt.xlabel("Ağaç sayısı (n_estimators)")
plt.ylabel("Test doğruluğu")
plt.title("Ağaç Sayısının Başarıya Etkisi")
plt.legend()
kaydet("09_agac_sayisi_etkisi.png")

# %% 12) HİPERPARAMETRE OPTİMİZASYONU (GridSearchCV)
# Bu bölümde ne yapıyoruz?
# - Random Forest için en iyi ayarları elle tahmin etmek yerine GridSearchCV ile bilgisayara denetiyoruz.
# - 12 kombinasyon deneniyor: ağaç sayısı (100, 200) × maksimum derinlik (3, 5, sınırsız) × min_samples_split (2, 5).
# - Her kombinasyon 5 katlı çapraz doğrulama ve ROC-AUC ile değerlendiriliyor, en iyisi seçiliyor.
# - Bu hücre birkaç saniye sürebilir. Takılmaması için n_jobs=1 kullanıyoruz.
#
parametre_izgarasi = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, None],
    "min_samples_split": [2, 5],
}
arama = GridSearchCV(RandomForestClassifier(random_state=RASTGELE, n_jobs=1),
                     parametre_izgarasi, cv=5, scoring="roc_auc", n_jobs=1)
arama.fit(X_egitim, y_egitim)
print("En iyi parametreler:", arama.best_params_)
print("En iyi CV ROC-AUC  :", round(arama.best_score_, 4))
print("Test doğruluğu     :", round(accuracy_score(y_test, arama.best_estimator_.predict(X_test)), 4))

# %% 13) TEK KARAR AĞACI GÖRSELİ (teori için)
# Bu bölümde ne yapıyoruz?
# - 3 derinlikli küçük bir karar ağacı eğitip çiziyoruz.
# - Her kutuda sorulan soru (örneğin "worst perimeter ≤ 112.8?"), Gini değeri, hasta sayısı ve çoğunluk sınıfı görünüyor.
# - Random Forest ve XGBoost'un içindeki her ağaç bu şekilde çalışır.
#
kucuk_agac = DecisionTreeClassifier(max_depth=3, random_state=RASTGELE).fit(X_egitim, y_egitim)
plt.figure(figsize=(14, 6))
plot_tree(kucuk_agac, feature_names=X.columns, class_names=["İyi huylu", "Habis"],
          filled=True, rounded=True, fontsize=8)
plt.title("Örnek Karar Ağacı (derinlik = 3)")
kaydet("10_ornek_karar_agaci.png")

# %% 14) YENİ / GÖRÜLMEMİŞ VERİ İLE CANLI TAHMİN
# Bu bölümde ne yapıyoruz?
# - tahmin_et() fonksiyonu tek bir hastanın ölçümlerini alıp Random Forest ve XGBoost'un kararını ve habis olma olasılığını yazdırıyor.
# - Üç hasta deniyoruz:
#   1. Test setinden gerçek bir hasta
#   2. Elle girilmiş, ölçümleri büyük (habis benzeri) bir hasta
#   3. Elle girilmiş, ölçümleri küçük (iyi huylu benzeri) bir hasta
# - Model bu hastaları eğitimde hiç görmedi. İstersek bir değeri değiştirip hücreyi tekrar çalıştırarak tahminin nasıl değiştiğini canlı gösterebiliriz.
#
# Ortalama bir hastadan başlayıp birkaç ölçümü değiştirerek sınıfta canlı deneme yapılabilir.
def tahmin_et(ornek, etiket="Yeni hasta"):
    ornek = pd.DataFrame([ornek], columns=X.columns)
    print(f"--- {etiket} ---")
    for ad in ["Random Forest", "XGBoost"]:
        p = modeller[ad].predict_proba(ornek)[0, 1]
        karar = "HABİS" if p >= 0.5 else "İYİ HUYLU"
        print(f"{ad:14s}: {karar:9s} (habis olasılığı: %{p * 100:.1f})")
    print()

# a) Test setinden rastgele bir gerçek örnek
i = np.random.RandomState(7).randint(len(X_test))
tahmin_et(X_test.iloc[i], f"Test örneği #{i} (gerçek etiket: {'HABİS' if y_test.iloc[i] else 'İYİ HUYLU'})")

# b) Elle girilen örnek: ortalama değerleri büyütülmüş (habis benzeri) hasta
elle = X.mean().copy()
elle["mean radius"] = 20.0
elle["mean perimeter"] = 130.0
elle["mean area"] = 1200.0
elle["worst radius"] = 25.0
elle["worst perimeter"] = 170.0
elle["worst concave points"] = 0.25
tahmin_et(elle, "Elle girilen örnek (büyük tümör ölçümleri)")

# c) Elle girilen örnek: ortalamadan küçük ölçümler
kucuk = X.mean().copy()
kucuk["mean radius"] = 11.0
kucuk["mean perimeter"] = 72.0
kucuk["mean area"] = 380.0
kucuk["worst radius"] = 12.5
kucuk["worst perimeter"] = 82.0
kucuk["worst concave points"] = 0.05
tahmin_et(kucuk, "Elle girilen örnek (küçük tümör ölçümleri)")

# %% 15) SONUÇ
# Bu bölümde ne yapıyoruz?
# - Üç modelin özet sonuç tablosunu tekrar yazdırıyoruz.
# - ROC-AUC'ye göre en iyi modeli ilan ediyoruz ve grafiklerin kaydedildiği klasörü gösteriyoruz.
# - Sonuç: Tek ağaç ~%93 doğruluk verirken ensemble yöntemler ~%97'ye çıkıyor.
#
print("ÖZET TABLO (Test Seti)")
print(sonuc)
print("\nEn iyi model (ROC-AUC'ye göre):", sonuc["ROC-AUC"].idxmax())
print("Grafikler kaydedildi ->", GRAFIK)

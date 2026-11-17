# Sunum Notları (Grup 5 - Random Forest & XGBoost)

## Teori (15-20 dk) - önerilen slayt akışı
1. Gözetimli öğrenme ve sınıflandırma problemi
2. Karar ağacı: bölünme, Gini / entropi, aşırı öğrenme sorunu
3. Ensemble fikri: Bagging ve Boosting farkı
4. Random Forest: bootstrap örnekleme + rastgele öznitelik seçimi + oy çoğunluğu; hiperparametreler (n_estimators, max_depth, min_samples_split)
5. XGBoost: gradient boosting, her ağaç önceki hatayı düzeltir; öğrenme hızı, düzenlileştirme (regularization), subsample
6. Avantajlar / kısıtlar (RF: paralel, kararlı, ama büyük ve yavaş tahmin; XGBoost: yüksek başarı, ama ayar gerektirir)
7. Veri seti: 569 örnek, 30 öznitelik (yarıçap, doku, çevre, alan, pürüzsüzlük... ortalama / standart hata / en kötü değer), eksik veri yok, ölçekleme gerekmez
8. Metrikler: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

## Olası sorular
- **Bagging ile boosting farkı?** Bagging ağaçları paralel ve bağımsız eğitir (varyansı azaltır); boosting sıralı eğitir, her ağaç öncekinin hatasına odaklanır (yanlılığı azaltır).
- **Neden ölçekleme yapmadık?** Ağaçlar eşik tabanlı bölünür, özniteliklerin ölçeğinden etkilenmez.
- **Bu veride neden Recall önemli?** Habis bir tümörü kaçırmak (yanlış negatif), iyi huylu birine habis demekten daha tehlikelidir.
- **Aşırı öğrenme nasıl azaltılır?** max_depth, min_samples_split, öğrenme hızı, subsample, çapraz doğrulama.
- **Alternatifler?** LightGBM, CatBoost, SVM, lojistik regresyon.
- **Veri seti küçük, sonuçlara güvenilir mi?** Bu yüzden 5 katlı çapraz doğrulama da gösterdik.

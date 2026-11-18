# Selma Bulut - Yapay Zeka Projesi (Grup 5)

**Konu:** Gözetimli Öğrenme - Karar Ağaçları & Ensemble Yöntemler
**Algoritma:** Random Forest & XGBoost Sınıflandırıcı
**Sunum tarihi:** 19 Kasım
**Veri seti:** Wisconsin Meme Kanseri Teşhis Veri Seti (569 örnek, 30 öznitelik, hedef: habis / iyi huylu).
scikit-learn ile birlikte gelir, internet gerekmez; `veri/meme_kanseri.csv` olarak da kaydedilir.

## Kurulum
```
pip install -r requirements.txt
```

## Çalıştırma
- **Jupyter (önerilen):** `jupyter notebook canli_kod.ipynb` -> hücreleri sırayla çalıştırın.
- **VS Code / PyCharm / Spyder:** `canli_kod.py` içindeki `# %%` hücrelerini tek tek çalıştırın.
- **Tek seferde:** `python canli_kod.py` (grafikler `grafikler/` klasörüne kaydedilir).

## Klasör yapısı
| Dosya / klasör | İçerik |
|---|---|
| `canli_kod.ipynb` / `canli_kod.py` | Canlı demo kodu (aynı içerik) |
| `veri/` | Veri seti CSV'si ve model sonuç tablosu |
| `grafikler/` | Kod çalışınca üretilen 10 grafik |
| `sunum_notlari.md` | Teori slaytları ve soru-cevap hazırlığı |

## Canlı kod akışı (yönergedeki sıra)
1. Veri yükleme -> 2. Keşifsel veri analizi (3 grafik) -> 3. Eğitim/test ayırma ->
4. Model eğitimi (Karar Ağacı, Random Forest, XGBoost) -> 5. Metrikler (Accuracy, Precision, Recall, F1, ROC-AUC) ->
6. Confusion Matrix ve ROC eğrisi -> 7. Çapraz doğrulama -> 8. Öznitelik önemleri ->
9. Hiperparametre etkisi ve GridSearchCV -> 10. Görülmemiş yeni örnekle canlı tahmin.

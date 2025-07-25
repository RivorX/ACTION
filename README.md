# 📈 Model predykcyjny kursu akcji

Projekt zawiera model trenowany na danych giełdowych oraz prostą aplikację webową do prezentacji wyników.

---

## 🚀 Trening modelu

Aby rozpocząć trening modelu, uruchom:

```bash
python start_training.py
```

Plik ten ładuje dane, trenuje model (np. Temporal Fusion Transformer) i zapisuje wyniki do pliku.

---

## 🖥️ Aplikacja webowa

Aby uruchomić aplikację Streamlit:

```bash
streamlit run app.py
```

Aplikacja umożliwia wizualizację prognoz oraz ocenę skuteczności modelu.

---

## ⚠️ Uwaga dotycząca PE i PB ratio

Biblioteka `yfinance` nie udostępnia **historycznych** wartości wskaźników fundamentalnych takich jak:

* PE ratio (Price to Earnings)
* PB ratio (Price to Book)

Możliwe jest jedynie pobranie ich **aktualnych wartości** z poziomu `Ticker().info`. 
---

## 📁 Struktura projektu

```
├── app.py                  # Aplikacja Streamlit
├── start_training.py       # Skrypt treningowy
├── model/                  # Pliki modelu i wag
├── data/                   # Dane rynkowe
└── README.md               # Dokumentacja
```

---

## 📌 Wymagania

* Python 3.9+
* streamlit
* yfinance
* pytorch / pytorch-lightning
* pandas, numpy, matplotlib

Zainstaluj zależności:

```bash
pip install -r requirements.txt
```


## Modele

- **gen3** – pełna wersja działającego modelu.
- **gen3mini** – zminiaturyzowana wersja modelu, lżejsza i szybsza w działaniu.
- **gen4mini** – wersja zredukowana, z mniejszą liczbą redundantnych cech (featureów), zoptymalizowana pod względem efektywności.

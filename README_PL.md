# Środowisko Obliczeniowe do Śledzenia Wzroku (Eye Tracking)

## O projekcie
Modułowe środowisko, umożliwiające testy różnych architektur sieci neuronowych, opierających się na współrzędnych punktów charakterystycznych twarzy uzyskanych za pomocą MediaPipe.
 Różne podejścia do predykcji kierunku spojrzenia znajdują się w dedykowanych gałęziach repozytorium.

## Funkcjonalności
- Testowanie różnych podejść do zadania śledzenia wzroku: regresja punktu spojrzenia oraz klasyfikacja obszaru skupienie wzroku (ROI);
- Rozbudowane narzędzia do analizy bazy danych w notatnikach Jupyter;
- Wsparcie dla wielu zbiorów danych;
- Łatwe przełączanie architektur sieci neuronowych wraz z hiperparametrami;
- Integracja procesu trenowania i ewaluacji wydajności z platformą Weights & Biases;
- Lokalne zapisywanie podsumowań wyników testów oraz wizualizacji.

## Wymagania
- Ubuntu (minimum 22.04)
- Karta graficzna NVIDIA z obsługą CUDA 12.x
- Python 3.10+

## Konfiguracja

```
curl -sSL https://install.python-poetry.org | python -
```
Instalacja zależności i utworzenie środowiska:

```
poetry install
```

Aktywacja środowiska:

```
poetry shell
```

Uruchomienie aplikacji:

```
poetry run python3 src/main.py
```

# Fase 3: Imputation (Imputazione e Valutazione Dati)

## Overview

La Fase 3 riguarda l'applicazione di algoritmi per stimare i dati mancanti (introdotti nella Fase 2) e la valutazione approfondita dei risultati ottenuti. I notebook e gli script relativi a questa fase sono organizzati all'interno della cartella `notebooks/Fase3/`.

## 3.1 Metodi di Imputazione

Il progetto esplora e confronta due macro-approcci per la ricostruzione dei dati mancanti:

### Approccio Standard / ML (`notebooks/Fase3/ML/`)
Utilizza tecniche statistiche e di Machine Learning consolidate:
- **Imputazione con Mediana** (`Mediana.ipynb`): Approccio di base in cui ogni valore mancante viene rimpiazzato con la mediana della rispettiva colonna.
- **MICE (Multiple Imputation by Chained Equations)** (`MICE.ipynb`): Algoritmo più sofisticato che stima i valori mancanti iterativamente, modellando ciascuna feature come variabile dipendente in funzione delle altre.

### Approccio Avanzato / DLLM (`notebooks/Fase3/DLLM/`)
Utilizza tecniche basate su Deep Learning e Modelli Linguistici (LLM/DLLM):
- **Mercury Imputation** (`DLLM_Mercury_Imputation.py`): Implementazione di modelli avanzati in grado di gestire contesti non lineari e catturare correlazioni complesse per la ricostruzione dei valori.

## 3.2 Valutazione e Confronto

L'efficacia delle strategie di imputazione viene valutata sotto due aspetti principali: la precisione diretta della ricostruzione e l'impatto sul risultato finale del task di classificazione (downstream task).

### Calcolo delle Metriche di Errore Diretto
Viene misurato lo scostamento puntuale tra i dati originari non corrotti e i dati ricostruiti:
- `Baseline_MSE_MAE_ML.ipynb` e `Baseline_MSE_MAE_DLLM.ipynb`: Valutano l'Errore Quadratico Medio (MSE) e l'Errore Assoluto Medio (MAE).

### Impatto sul Modello Predittivo (Downstream Task)
L'obiettivo ultimo del dataset è addestrare un classificatore per la predizione del Rischio di Credito:
- `XGBoost_ML.ipynb` e `XGBoost_DLLM.ipynb`: Addestramento di modelli di classificazione basati su XGBoost utilizzando i dataset imputati. Questo step valuta l'impatto qualitativo dell'imputazione sulle performance predittive finali in uno scenario reale di classificazione.

### Analisi Comparativa Finale
- `Compare_Results.ipynb`: Notebook di sintesi che aggrega tutte le metriche per offrire un confronto conclusivo tra i metodi classici (ML) e i metodi avanzati (DLLM).
- `Imputed_Originals_differences.ipynb`: Analisi statistica e visiva approfondita delle distribuzioni e delle differenze puntuali tra i valori reali e quelli stimati, per comprendere l'eventuale bias o distorsione introdotta dall'imputazione.

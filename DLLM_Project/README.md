# DLLM Project - Credit Risk Analysis

Preprocessing pipeline per il dataset HELOC (Home Equity Line of Credit) destinato al training di modelli di Deep Learning per la stima del rischio di credito.

## Struttura del Progetto

```
DLLM_Project/
├── data/
│   ├── raw/           # Dataset originale
│   └── processed/     # Dataset cleaned e ready for training
├── notebooks/         # Jupyter notebooks per preprocessing e analysis
├── docs/              # Documentazione del preprocessing
├── config/            # Configurazioni e mappings
└── README.md
```

## Dataset

- **Input**: `data/raw/heloc_dataset.csv` (10,459 righe, 24 colonne)
- **Output**: `data/processed/heloc_dataset_cleaned.csv` (9,861 righe, pulito)

## Preprocessing

Il notebook `notebooks/PreProcessing.ipynb` implementa:

1. **Rimozione di righe con -9** (588 righe = 5.62%)
2. **Sostituzione di -7** (206 valori con stringhe semantiche)
3. **Sostituzione di -8** (11,072 valori con stringhe semantiche)

Vedi `docs/PREPROCESSING.md` per i dettagli completi.

## Dataset Cleaning Statistics

- Righe originali: 10,459
- Righe dopo pulizia: 9,861
- Righe eliminate: 598 (5.72%)
- Special values rimossi: 11,866 (206 -7 + 11,072 -8 + 588 -9)

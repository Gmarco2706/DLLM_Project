#!/usr/bin/env python3

import random
import requests
import re
import csv
import time
import sys
import os
import shutil
import logging
import traceback
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# 6 Chiavi API Mercury — una per ogni parte del dataset
API_KEYS = [
    "sk_0dd4042c2ccce1b874bd80ebc3f214eb",  # Part 1
    "sk_01644b633cc334645807937b4eb8ac06",  # Part 2
    "sk_d67325388d4204e7306bd1e9ab21e028",  # Part 3
    "sk_f6e7396a1319c6347f216a250b5d477c",   # Part 4
    "sk_ebe52dcfbe9b7973f19d8b461a3caed9",   # Part 5
    "sk_7537ae54627fc43115a0b949eea0f572",  # Part 6
]

# Percorsi base del progetto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "Fase2" / "DataCorruption" / "heloc_DLLM"
DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "Fase3" / "Imputated_DLLM"
PARTS_OUTPUT_DIR = DATA_OUTPUT_DIR / "DiscriminativeParts"

# Dataset da processare: (meccanismo, percentuale)
DATASETS = [
    ("MCAR", "10"),
    ("MCAR", "25"),
    ("MCAR", "40"),
    ("MAR", "10"),
    ("MAR", "25"),
    ("MAR", "40"),
    ("MNAR", "10"),
    ("MNAR", "25"),
    ("MNAR", "40"),
]

# Parametri di split e parallelismo
NUM_SPLITS = 6
MAX_WORKERS = NUM_SPLITS  # Un worker per parte = 6 worker

# Se True, rielabora TUTTI i dataset da zero (cancella output esistenti).
# Se False, salta i dataset/parti già completati (modalità resume).
FORCE_REPROCESS = False

# Parametri modello Mercury
MODEL = "mercury-2"
TEMPERATURE = 0.2
API_URL = "https://api.inceptionlabs.ai/v1/chat/completions"

# Parametri retry
MAX_CONTENT_NULL_RETRIES = 5   # Retry per content=null prima di rinunciare
MAX_NETWORK_RETRIES = 10       # Retry per errori di rete
MAX_QUOTA_ERRORS = 10          # Retry per HTTP 402 (quota)
# Rate limit (429) e server error (5xx): retry INFINITI con backoff, cap a 60s

# Parametri per il cleanup pass finale
MAX_CLEANUP_PASSES = 3         # Numero massimo di passaggi di pulizia

# Campi HELOC
FIELDS = [
    "ExternalRiskEstimate", "MSinceOldestTradeOpen",
    "MSinceMostRecentTradeOpen", "AverageMInFile", "NumSatisfactoryTrades",
    "NumTrades60Ever2DerogPubRec", "NumTrades90Ever2DerogPubRec",
    "PercentTradesNeverDelq", "MSinceMostRecentDelq", "MaxDelq2PublicRecLast12M",
    "MaxDelqEver", "NumTotalTrades", "NumTradesOpeninLast12M",
    "PercentInstallTrades", "MSinceMostRecentInqexcl7days", "NumInqLast6M",
    "NumInqLast6Mexcl7days", "NetFractionRevolvingBurden",
    "NetFractionInstallBurden", "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance", "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance"
]

# Valori stringa da trattare come "da imputare" (devono diventare numerici)
STRING_VALUES = {
    "Never Had Delinquency",
    "New to Credit System",
    "No Bank Revolving",
    "No Credit Inquiry",
    "No Inquiry Record",
    "No Installment Burden",
    "No Installment Loans",
    "No Payment History",
    "No Revolving Accounts",
    "No Revolving Burden",
    "No Valid Credit Accounts",
}

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("DLLM_Imputation")

# ============================================================================
# FUNZIONI DI SUPPORTO
# ============================================================================

def _is_missing(val) -> bool:
    """Controlla se un valore è vuoto/nullo."""
    if val is None:
        return True
    s = str(val).strip()
    return s in {"", "???", "????", "nan", "NaN", "None", "null", "NULL"}


def _is_string_value(val) -> bool:
    """Controlla se un valore è una delle stringhe da imputare."""
    if val is None:
        return False
    return str(val).strip() in STRING_VALUES


def _needs_replacement(val) -> bool:
    """Controlla se un valore deve essere imputato (mancante O stringa)."""
    return _is_missing(val) or _is_string_value(val)


def needs_imputation(row: dict) -> bool:
    """Controlla se una riga ha almeno un campo da imputare (mancante o stringa)."""
    for field in FIELDS:
        val = row.get(field, "")
        if _needs_replacement(val):
            return True
    return False


def get_missing_fields(row: dict) -> list:
    """Restituisce i campi da imputare (mancanti o con stringhe)."""
    return [f for f in FIELDS if _needs_replacement(row.get(f, ""))]


# ============================================================================
# FEW-SHOT POOL
# ============================================================================
CLEAN_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "Fase2" / "SplitDataset" / "Split_DLLM" / "heloc_DLLM_imputation_train.csv"

good_pool = []
bad_pool = []

def _load_few_shot_pool() -> None:
    """Carica il pool di esempi few-shot dal dataset clean di training."""
    global good_pool, bad_pool
    if not CLEAN_TRAIN_PATH.exists():
        logger.warning(f"Dataset clean non trovato in {CLEAN_TRAIN_PATH}. Few-shot disabilitato.")
        return
    try:
        df_clean = pd.read_csv(CLEAN_TRAIN_PATH)
        _good, _bad = [], []
        for _, r in df_clean.iterrows():
            row_dict = r.to_dict()
            if any(_is_missing(row_dict.get(f, "")) or _is_string_value(row_dict.get(f, "")) for f in FIELDS):
                continue  # salta righe con valori da imputare
            risk = row_dict.get('RiskPerformance', '')
            row_prompt = {k: v for k, v in row_dict.items() if k != 'RiskPerformance'}
            if risk == 'Good':
                _good.append(row_prompt)
            elif risk == 'Bad':
                _bad.append(row_prompt)
        good_pool, bad_pool = _good, _bad
        logger.info(f"Few-shot pool caricato: {len(good_pool)} Good completi, {len(bad_pool)} Bad completi.")
    except Exception as e:
        logger.error(f"Errore caricamento few-shot pool: {e}")

def build_row_string_optimized(row: dict) -> str:
    """Costruisce una stringa compatta della riga per il prompt."""
    parts = []
    for f in FIELDS:
        v = str(row.get(f, "")).strip()
        if _needs_replacement(v):
            v = "?"
        parts.append(f"{f}={v}")
    return ",".join(parts)


def build_prompt_optimized(row_str: str, missing_fields: list, few_shot_str: str = "") -> str:
    """Costruisce il prompt per il modello Mercury."""
    if missing_fields:
        missing_fmt = ",".join(f"{f}=?" for f in missing_fields)
        if few_shot_str:
            example = f"Here are 4 examples of valid values from the dataset (showing only relevant fields):\n{few_shot_str}\n\nNow impute the following row:\n"
        else:
            example = (
                "Example:\n"
                "Input: ExternalRiskEstimate=86,MSinceOldestTradeOpen=?,"
                "AverageMInFile=?,NumTotalTrades=12\n"
                "Output: MSinceOldestTradeOpen=219,AverageMInFile=97\n"
            )
    else:
        missing_fmt = ""
        example = ""

    return (f"Impute the missing values (marked with ?) in this HELOC credit row.\n"
            f"All values MUST be numeric (integers or decimals). Do not return strings or text.\n"
            f"Do not change existing values.\n"
            f"Return only the imputed fields: {missing_fmt}\n\n"
            f"{example}"
            f"Row: {row_str}")


def extract_pairs(text: str) -> dict:
    """Estrae coppie chiave=valore dalla risposta del modello."""
    pairs = re.findall(r'(\w+)=([^,\n]+)', text)
    return {k.strip(): v.strip().rstrip(".") for k, v in pairs}


class QuotaExhaustedException(Exception):
    """Eccezione lanciata quando la quota API è esaurita."""
    pass


# ============================================================================
# API KEY POOL (thread-safe, con fallback dinamico)
# ============================================================================

class APIKeyPool:
    """
    Pool thread-safe di chiavi API.
    Quando una chiave esaurisce la quota, viene marcata come 'exhausted'
    e i worker possono richiedere una chiave alternativa ancora attiva.
    """

    def __init__(self, keys: list):
        self._lock = threading.Lock()
        self._all_keys = list(keys)
        self._alive_keys = set(range(len(keys)))   # indici delle chiavi attive
        self._exhausted_keys = set()                # indici delle chiavi esaurite

    def get_key(self, preferred_idx: int) -> str:
        """Restituisce la chiave preferita se ancora attiva, altrimenti None."""
        with self._lock:
            if preferred_idx in self._alive_keys:
                return self._all_keys[preferred_idx]
        return None

    def mark_exhausted(self, key: str) -> None:
        """Marca una chiave come esaurita."""
        with self._lock:
            for i, k in enumerate(self._all_keys):
                if k == key:
                    self._alive_keys.discard(i)
                    self._exhausted_keys.add(i)
                    logger.warning(f"🔑 Chiave API #{i+1} marcata come esaurita. "
                                 f"Chiavi attive rimaste: {len(self._alive_keys)}/{len(self._all_keys)}")
                    break

    def get_fallback_key(self, exclude_key: str = None) -> str | None:
        """
        Restituisce una chiave ancora attiva diversa da exclude_key.
        Se non ci sono chiavi disponibili, restituisce None.
        """
        with self._lock:
            for i in self._alive_keys:
                if self._all_keys[i] != exclude_key:
                    return self._all_keys[i]
            return None

    def get_any_alive_key(self) -> str | None:
        """Restituisce una qualsiasi chiave ancora attiva."""
        with self._lock:
            for i in self._alive_keys:
                return self._all_keys[i]
            return None

    def has_alive_keys(self) -> bool:
        """Controlla se ci sono ancora chiavi attive."""
        with self._lock:
            return len(self._alive_keys) > 0

    def alive_count(self) -> int:
        """Restituisce il numero di chiavi ancora attive."""
        with self._lock:
            return len(self._alive_keys)

    def get_status_summary(self) -> str:
        """Restituisce un riepilogo dello stato delle chiavi."""
        with self._lock:
            alive = sorted(i + 1 for i in self._alive_keys)
            exhausted = sorted(i + 1 for i in self._exhausted_keys)
            return (f"Attive: {alive} ({len(alive)}/{len(self._all_keys)}) | "
                    f"Esaurite: {exhausted}")


# ============================================================================
# TEST API KEYS
# ============================================================================

def test_api_keys(keys: list) -> list:
    """
    Testa tutte le chiavi API con una richiesta minimale.
    Restituisce la lista degli indici delle chiavi funzionanti.
    """
    logger.info(f"\n{'='*60}")
    logger.info("🔑 TEST CHIAVI API")
    logger.info(f"{'='*60}")

    working_keys = []
    test_prompt = "Reply with just the number 1."

    for i, key in enumerate(keys):
        key_label = f"Chiave #{i+1}"
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL,
                    "temperature": 0.0,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "max_tokens": 5
                },
                timeout=30
            )

            if response.status_code == 200:
                try:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if content is not None:
                        logger.info(f"  ✅ {key_label}: OK (risposta: '{content.strip()[:20]}')")
                        working_keys.append(i)
                    else:
                        logger.warning(f"  ⚠️  {key_label}: HTTP 200 ma content=null")
                        working_keys.append(i)  # potrebbe funzionare comunque
                except (KeyError, IndexError, ValueError) as e:
                    logger.warning(f"  ⚠️  {key_label}: risposta malformata ({e})")
                    working_keys.append(i)  # potrebbe funzionare
            elif response.status_code == 402:
                logger.error(f"  ❌ {key_label}: QUOTA ESAURITA (HTTP 402)")
            elif response.status_code == 401:
                logger.error(f"  ❌ {key_label}: NON AUTORIZZATA (HTTP 401)")
            elif response.status_code == 429:
                logger.warning(f"  ⚠️  {key_label}: Rate limit (HTTP 429) — probabilmente funzionante")
                working_keys.append(i)
            else:
                err_msg = response.text[:100]
                logger.warning(f"  ⚠️  {key_label}: HTTP {response.status_code} — {err_msg}")
                working_keys.append(i)  # non è 402 o 401, potrebbe funzionare

        except requests.exceptions.Timeout:
            logger.warning(f"  ⚠️  {key_label}: Timeout (30s) — potrebbe funzionare")
            working_keys.append(i)
        except requests.exceptions.RequestException as e:
            logger.error(f"  ❌ {key_label}: Errore di rete — {e}")

        # Piccola pausa tra i test per evitare rate limit
        time.sleep(0.5)

    logger.info(f"\n  Risultato: {len(working_keys)}/{len(keys)} chiavi funzionanti")
    if working_keys:
        logger.info(f"  Chiavi OK: {[i+1 for i in working_keys]}")
    else:
        logger.error("  ❌ NESSUNA CHIAVE FUNZIONANTE! Impossibile procedere.")

    return working_keys


# ============================================================================
# SPLIT DATASET
# ============================================================================

def split_dataset(input_path: Path, num_splits: int = NUM_SPLITS) -> list:
    """
    Divide il dataset in `num_splits` parti sequenziali.
    Restituisce la lista dei path delle parti create.
    """
    logger.info(f"Splitting dataset: {input_path.name} → {num_splits} parti")
    df = pd.read_csv(input_path)

    splits_indices = np.array_split(df.index, num_splits)
    splits = [df.loc[idx] for idx in splits_indices]

    output_files = []
    base_name = input_path.stem
    # Scrive le parti nella directory PARTS_OUTPUT_DIR invece che accanto all'input
    PARTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, split_df in enumerate(splits):
        output_path = PARTS_OUTPUT_DIR / f"{base_name}_part{i+1}.csv"
        split_df.to_csv(output_path, index=False)
        output_files.append(output_path)
        logger.info(f"  Parte {i+1}: {len(split_df)} righe → {output_path.name}")

    return output_files


# ============================================================================
# CONTEGGIO RIGHE GIÀ PROCESSATE (per resume)
# ============================================================================

def count_processed_rows(output_csv: Path) -> int:
    """Conta le righe già scritte nel file di output (per il resume)."""
    if not output_csv.exists():
        return 0
    try:
        with open(output_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)
    except Exception:
        return 0


# ============================================================================
# IMPUTATION DI UNA SINGOLA RIGA (ultra-resiliente)
# ============================================================================

def impute_row(row: dict, api_key: str, part_label: str) -> dict:
    """
    Imputa i valori mancanti di una singola riga usando Mercury API.
    
    Strategia di retry:
    - Rate limit (429): retry INFINITI con backoff esponenziale (cap 60s)
    - Server error (5xx): retry INFINITI con backoff esponenziale (cap 60s)
    - content=null: retry fino a MAX_CONTENT_NULL_RETRIES
    - Errore di rete: retry fino a MAX_NETWORK_RETRIES
    - Quota (402): retry fino a MAX_QUOTA_ERRORS, poi QuotaExhaustedException
    """
    missing = get_missing_fields(row)
    if not missing:
        return row

    few_shot_str = ""
    if len(good_pool) >= 2 and len(bad_pool) >= 2:
        sampled_good = random.sample(good_pool, 2)
        sampled_bad = random.sample(bad_pool, 2)
        sampled = sampled_good + sampled_bad
        random.shuffle(sampled)
        
        # Seleziona solo i campi mancanti + un paio di ancore per dare contesto
        key_fields = ["ExternalRiskEstimate", "NumTotalTrades"]
        fields_to_show = []
        for f in key_fields:
            if f not in missing:
                fields_to_show.append(f)
        fields_to_show.extend(missing)
        
        lines = []
        for i, s_row in enumerate(sampled):
            s_str = ",".join(f"{f}={s_row.get(f,'')}" for f in fields_to_show if f in s_row)
            lines.append(f"Example {i+1}: {s_str}")
        few_shot_str = "\n".join(lines)

    row_str = build_row_string_optimized(row)
    prompt = build_prompt_optimized(row_str, missing, few_shot_str)

    network_retries = 0
    quota_errors = 0
    content_null_retries = 0
    consecutive_errors = 0  # Per backoff generico

    while True:
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=90
            )

            try:
                data = response.json()
            except ValueError:
                data = {"error": response.text}

            # --- Rate limiting (429): retry INFINITI ---
            if response.status_code == 429:
                consecutive_errors += 1
                wait_time = min(5 * consecutive_errors, 60)
                logger.warning(f"[{part_label}] Rate limit (429), attendo {wait_time}s...")
                time.sleep(wait_time)
                continue

            # --- Quota esaurita (402) ---
            if response.status_code == 402:
                quota_errors += 1
                if quota_errors >= MAX_QUOTA_ERRORS:
                    logger.error(f"[{part_label}] Quota esaurita (HTTP 402 x{quota_errors})!")
                    raise QuotaExhaustedException(f"Quota esaurita dopo {quota_errors} errori 402")
                logger.warning(f"[{part_label}] HTTP 402 ({quota_errors}/{MAX_QUOTA_ERRORS}), attendo 30s...")
                time.sleep(30)
                continue

            # --- Errore server (5xx): retry INFINITI ---
            if response.status_code >= 500:
                consecutive_errors += 1
                wait_time = min(5 * consecutive_errors, 60)
                err_msg = str(data.get('error', response.text))[:100]
                logger.warning(f"[{part_label}] Server error HTTP {response.status_code}: {err_msg}. "
                             f"Attendo {wait_time}s...")
                time.sleep(wait_time)
                continue

            # --- Rate limit keyword nel messaggio (non 429) ---
            err_text = str(data.get("error", "")).lower()
            if any(k in err_text for k in ["rate limit", "too many requests", "slow down"]):
                consecutive_errors += 1
                wait_time = min(5 * consecutive_errors, 60)
                logger.warning(f"[{part_label}] Rate limit (body), attendo {wait_time}s...")
                time.sleep(wait_time)
                continue

            # --- Errore non-success generico (4xx diverso da 402/429) ---
            if response.status_code != 200:
                err_msg = str(data.get('error', response.text))[:200]
                logger.error(f"[{part_label}] Errore API HTTP {response.status_code}: {err_msg}")
                return row

            # --- Successo (200) ma con errore nel body ---
            if "error" in data and data["error"]:
                consecutive_errors += 1
                err_msg = str(data['error'])[:200]
                wait_time = min(5 * consecutive_errors, 60)
                logger.warning(f"[{part_label}] Errore nel body (HTTP 200): {err_msg}. "
                             f"Attendo {wait_time}s...")
                time.sleep(wait_time)
                continue

            # --- Successo: estrai risposta ---
            # Reset consecutive errors counter on success
            consecutive_errors = 0

            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                logger.error(f"[{part_label}] Risposta malformata: {e}. Data: {str(data)[:200]}")
                return row

            # --- content=null: RIPROVA ---
            if content is None:
                content_null_retries += 1
                if content_null_retries > MAX_CONTENT_NULL_RETRIES:
                    logger.warning(f"[{part_label}] content=null x{content_null_retries}. "
                                 f"Rinuncio, salvo riga originale.")
                    return row
                wait_time = 3 * content_null_retries
                logger.warning(f"[{part_label}] content=null, riprovo tra {wait_time}s "
                             f"({content_null_retries}/{MAX_CONTENT_NULL_RETRIES})...")
                time.sleep(wait_time)
                continue

            text = content.strip()
            imputed = extract_pairs(text)

            result = dict(row)
            for field in missing:
                if field in imputed:
                    val = imputed[field]
                    # Verifica che il valore non sia ancora una stringa non numerica
                    if val and val.strip() in STRING_VALUES:
                        logger.warning(f"[{part_label}] Modello ha restituito stringa per "
                                     f"{field}: '{val}'. Ignoro.")
                        continue
                    result[field] = val

            return result

        except QuotaExhaustedException:
            raise  # propaga l'eccezione
        except requests.exceptions.RequestException as e:
            network_retries += 1
            if network_retries > MAX_NETWORK_RETRIES:
                logger.error(f"[{part_label}] Troppe richieste di rete fallite ({network_retries}): {e}")
                return row
            wait_time = min(5 * network_retries, 30)
            logger.warning(f"[{part_label}] Errore di rete: {e}. Riprovo tra {wait_time}s "
                         f"({network_retries}/{MAX_NETWORK_RETRIES})...")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"[{part_label}] Errore inatteso in impute_row: {e}")
            logger.debug(traceback.format_exc())
            return row


# ============================================================================
# IMPUTATION DI UNA PARTE (worker per thread, con fallback chiavi)
# ============================================================================

def impute_part(input_csv: Path, output_csv: Path, api_key: str,
                part_num: int, dataset_label: str,
                key_pool: APIKeyPool = None) -> bool:
    """
    Imputa un singolo file parte.
    Supporta il resume: se output_csv esiste già parzialmente, riprende.
    Se key_pool è fornito, in caso di quota esaurita tenta con un'altra chiave.

    Returns:
        True se l'imputazione è completata con successo, False se interrotta.
    """
    part_label = f"{dataset_label}/part{part_num}"
    current_key = api_key

    # Carica input
    try:
        with open(input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames
    except Exception as e:
        logger.error(f"[{part_label}] Impossibile leggere {input_csv}: {e}")
        return False

    total_rows = len(rows)
    if total_rows == 0:
        logger.warning(f"[{part_label}] File input vuoto. Skip.")
        return True

    # Resume: conta righe già processate
    already_processed = count_processed_rows(output_csv)
    if already_processed >= total_rows:
        logger.info(f"[{part_label}] Già completata ({already_processed}/{total_rows} righe). Skip.")
        return True

    if already_processed > 0:
        logger.info(f"[{part_label}] Resume da riga {already_processed + 1}/{total_rows}")

    # Apri file in modalità append
    write_header = not output_csv.exists() or already_processed == 0
    mode = "w" if write_header else "a"

    try:
        with open(output_csv, mode, newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                out_f.flush()

            for idx, row in enumerate(rows):
                if idx < already_processed:
                    continue  # salta righe già processate

                if needs_imputation(row):
                    # Tenta l'imputazione, con fallback su altre chiavi
                    result = None
                    while result is None:
                        try:
                            result = impute_row(row, current_key, part_label)
                        except QuotaExhaustedException:
                            # Marca la chiave come esaurita nel pool
                            if key_pool:
                                key_pool.mark_exhausted(current_key)
                                fallback = key_pool.get_fallback_key(exclude_key=current_key)
                                if fallback:
                                    logger.warning(
                                        f"[{part_label}] 🔄 Quota esaurita, passo a chiave di fallback. "
                                        f"Stato pool: {key_pool.get_status_summary()}"
                                    )
                                    current_key = fallback
                                    continue  # riprova con la nuova chiave
                                else:
                                    logger.error(
                                        f"[{part_label}] ❌ Tutte le chiavi API esaurite! "
                                        f"Impossibile continuare."
                                    )
                                    return False
                            else:
                                logger.error(f"[{part_label}] ❌ Quota esaurita (nessun pool disponibile).")
                                return False
                else:
                    result = row

                writer.writerow(result)

                # Flush e log progresso ogni 50 righe o alla fine
                current = idx + 1
                if current % 50 == 0 or current == total_rows:
                    out_f.flush()
                    logger.info(f"[{part_label}] Progresso: {current}/{total_rows} righe")

        logger.info(f"[{part_label}] ✅ Completata! ({total_rows} righe)")
        return True

    except QuotaExhaustedException:
        # Quota esaurita fuori dal loop interno (non dovrebbe succedere, ma per sicurezza)
        if key_pool:
            key_pool.mark_exhausted(current_key)
        logger.error(f"[{part_label}] ❌ Interrotta: quota API esaurita.")
        return False
    except Exception as e:
        logger.error(f"[{part_label}] ❌ Errore imprevisto: {e}")
        logger.debug(traceback.format_exc())
        return False


# ============================================================================
# CONCATENAZIONE
# ============================================================================

def concatenate_parts(part_files: list, output_path: Path):
    """Concatena le parti imputate in un unico dataset."""
    logger.info(f"Concatenando {len(part_files)} parti → {output_path.name}")
    dfs = [pd.read_csv(f) for f in part_files]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_path, index=False)
    logger.info(f"  Dataset finale: {len(combined)} righe salvate in {output_path.name}")
    return combined


# ============================================================================
# PULIZIA FILE INTERMEDI
# ============================================================================

def cleanup_intermediate_files(part_input_files: list, part_output_files: list,
                                dataset_label: str):
    """
    Elimina i file intermedi (parti split dell'input e parti imputate di output)
    dopo la concatenazione riuscita.
    """
    logger.info(f"[{dataset_label}] 🗑️  Pulizia file intermedi...")
    deleted_count = 0

    # Elimina le parti split dell'input (create da split_dataset)
    for f in part_input_files:
        if f.exists():
            try:
                f.unlink()
                logger.info(f"[{dataset_label}]   Eliminato split input: {f.name}")
                deleted_count += 1
            except OSError as e:
                logger.warning(f"[{dataset_label}]   Impossibile eliminare {f.name}: {e}")

    # Elimina le parti imputate di output (nella directory DiscriminativeParts)
    for f in part_output_files:
        if f.exists():
            try:
                f.unlink()
                logger.info(f"[{dataset_label}]   Eliminato part output: {f.name}")
                deleted_count += 1
            except OSError as e:
                logger.warning(f"[{dataset_label}]   Impossibile eliminare {f.name}: {e}")

    logger.info(f"[{dataset_label}] 🗑️  Pulizia completata: {deleted_count} file eliminati")


# ============================================================================
# CLEANUP PASS FINALE (ri-imputa righe incomplete)
# ============================================================================

def cleanup_dataset(csv_path: Path, api_key: str, dataset_label: str) -> int:
    """
    Scansiona il dataset finale e ri-imputa le righe che hanno ancora
    campi mancanti o stringhe da rimpiazzare.

    Returns:
        Numero di righe corrette.
    """
    logger.info(f"[{dataset_label}] 🧹 Cleanup pass: scansione righe incomplete...")

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    rows_fixed = 0
    rows_still_broken = 0

    for idx in range(len(df)):
        row = df.iloc[idx].to_dict()

        if not needs_imputation(row):
            continue

        missing = get_missing_fields(row)
        logger.info(f"[{dataset_label}] Cleanup riga {idx+1}: {len(missing)} campi da imputare "
                   f"({', '.join(missing[:5])}{'...' if len(missing) > 5 else ''})")

        result = impute_row(row, api_key, f"{dataset_label}/cleanup")

        # Verifica se l'imputazione ha funzionato
        remaining = get_missing_fields(result)
        if len(remaining) < len(missing):
            for field in FIELDS:
                df.at[idx, field] = str(result.get(field, df.at[idx, field]))
            rows_fixed += 1
            if remaining:
                rows_still_broken += 1
                logger.warning(f"[{dataset_label}] Riga {idx+1}: imputati {len(missing)-len(remaining)}/{len(missing)} "
                             f"campi, restano {len(remaining)}")
        else:
            rows_still_broken += 1
            logger.warning(f"[{dataset_label}] Riga {idx+1}: nessun miglioramento")

    if rows_fixed > 0:
        df.to_csv(csv_path, index=False)
        logger.info(f"[{dataset_label}] 🧹 Cleanup completato: {rows_fixed} righe migliorate, "
                   f"{rows_still_broken} ancora incomplete")
    else:
        logger.info(f"[{dataset_label}] 🧹 Cleanup: nessuna riga da correggere")

    return rows_still_broken


# ============================================================================
# PIPELINE PER UN SINGOLO DATASET
# ============================================================================

def process_dataset(mechanism: str, percentage: str) -> bool:
    """
    Pipeline completa per un singolo dataset:
    split → impute (parallel) → concatenate → cleanup.

    Returns:
        True se il dataset è stato processato con successo.
    """
    dataset_label = f"{mechanism}_{percentage}"
    input_file = DATA_INPUT_DIR / f"heloc_DLLM_imputation_test_corrupted_{dataset_label}.csv"

    if not input_file.exists():
        logger.error(f"[{dataset_label}] File input non trovato: {input_file}")
        return False

    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING: {dataset_label}")
    logger.info(f"{'='*60}")

    # Output finale
    final_output = DATA_OUTPUT_DIR / f"heloc_DLLM_discriminative_train_{dataset_label}.csv"

    # Lista dei file di output delle parti
    part_output_files = []
    for i in range(NUM_SPLITS):
        part_output = PARTS_OUTPUT_DIR / f"heloc_DLLM_discriminative_train_{dataset_label}_part{i+1}.csv"
        part_output_files.append(part_output)

    if FORCE_REPROCESS:
        # Cancella output esistenti per forzare la rielaborazione
        if final_output.exists():
            final_output.unlink()
            logger.info(f"[{dataset_label}] Cancellato output finale precedente (FORCE_REPROCESS=True)")
        for pf in part_output_files:
            if pf.exists():
                pf.unlink()
                logger.info(f"[{dataset_label}] Cancellata parte precedente: {pf.name}")
    else:
        # Modalità resume: controlla se il dataset è già stato completato
        if final_output.exists():
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(final_output)
            if len(output_df) == len(input_df):
                logger.info(f"[{dataset_label}] Dataset già concatenato ({len(output_df)} righe). Eseguo cleanup...")
                # Esegui comunque il cleanup per sistemare righe incomplete
                cleanup_key = API_KEYS[0]
                for pass_num in range(1, MAX_CLEANUP_PASSES + 1):
                    logger.info(f"[{dataset_label}] 🧹 Cleanup pass {pass_num}/{MAX_CLEANUP_PASSES}")
                    still_broken = cleanup_dataset(final_output, cleanup_key, dataset_label)
                    if still_broken == 0:
                        logger.info(f"[{dataset_label}] ✅ Tutte le righe imputate correttamente!")
                        break
                    else:
                        logger.warning(f"[{dataset_label}] Ancora {still_broken} righe incomplete dopo pass {pass_num}")
                return True
            else:
                logger.warning(f"[{dataset_label}] Dataset finale esiste ma con righe diverse "
                             f"({len(output_df)} vs {len(input_df)}). Ri-elaboro.")

    # --- STEP 1: SPLIT ---
    part_input_files = split_dataset(input_file, NUM_SPLITS)

    # --- STEP 2: IMPUTE (in parallelo, con pool di chiavi condiviso) ---
    PARTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Crea il pool condiviso di chiavi API
    key_pool = APIKeyPool(API_KEYS)

    # Determina quali parti devono ancora essere processate
    parts_to_process = []
    for i in range(NUM_SPLITS):
        with open(part_input_files[i], encoding="utf-8") as _f:
            input_rows = sum(1 for _ in _f) - 1  # -1 per header
        processed_rows = count_processed_rows(part_output_files[i])
        if processed_rows < input_rows:
            parts_to_process.append(i)
        else:
            logger.info(f"[{dataset_label}/part{i+1}] Già completata. Skip.")

    if not parts_to_process:
        logger.info(f"[{dataset_label}] Tutte le parti sono già imputate. Procedo alla concatenazione.")
    else:
        logger.info(f"[{dataset_label}] Parti da processare: {[p+1 for p in parts_to_process]}")
        logger.info(f"[{dataset_label}] Stato pool chiavi: {key_pool.get_status_summary()}")

        num_keys = len(API_KEYS)
        all_success = True

        # Ogni parte usa la propria chiave dedicata, con fallback dal pool
        effective_workers = min(MAX_WORKERS, len(parts_to_process))
        logger.info(f"[{dataset_label}] Lancio {effective_workers} worker paralleli "
                   f"({key_pool.alive_count()} chiavi API attive)")

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {}
            for part_idx in parts_to_process:
                # Assegnazione chiave iniziale: part_idx % num_keys
                # Se la chiave è esaurita, il worker userà il pool per ottenerne un'altra
                key_idx = part_idx % num_keys
                api_key = key_pool.get_key(key_idx)
                if api_key is None:
                    # Chiave preferita già esaurita, prendi qualsiasi chiave attiva
                    api_key = key_pool.get_any_alive_key()
                    if api_key is None:
                        logger.error(f"[{dataset_label}] ❌ Nessuna chiave API disponibile!")
                        return False
                    logger.warning(f"[{dataset_label}/part{part_idx+1}] Chiave #{key_idx+1} esaurita, "
                                 f"uso chiave di fallback")
                else:
                    logger.info(f"[{dataset_label}/part{part_idx+1}] Usa chiave API #{key_idx+1}")

                future = executor.submit(
                    impute_part,
                    part_input_files[part_idx],
                    part_output_files[part_idx],
                    api_key,
                    part_idx + 1,
                    dataset_label,
                    key_pool
                )
                futures[future] = part_idx + 1

            for future in as_completed(futures):
                part_num = futures[future]
                try:
                    success = future.result()
                    if not success:
                        all_success = False
                        logger.error(f"[{dataset_label}/part{part_num}] Imputazione fallita.")
                except Exception as e:
                    all_success = False
                    logger.error(f"[{dataset_label}/part{part_num}] Eccezione: {e}")

        logger.info(f"[{dataset_label}] Stato finale pool chiavi: {key_pool.get_status_summary()}")

        if not all_success:
            logger.error(f"[{dataset_label}] ⚠️  Alcune parti hanno avuto errori. "
                        f"Il dataset finale potrebbe essere incompleto.")
            logger.info(f"[{dataset_label}] Puoi ri-eseguire lo script per tentare il resume.")
            return False

    # --- STEP 3: CONCATENATE ---
    # Verifica che tutte le parti esistano
    missing_parts = [f for f in part_output_files if not f.exists()]
    if missing_parts:
        logger.error(f"[{dataset_label}] Parti mancanti per la concatenazione: "
                    f"{[f.name for f in missing_parts]}")
        return False

    concatenate_parts(part_output_files, final_output)

    # Verifica integrità (numero righe)
    input_df = pd.read_csv(input_file)
    output_df = pd.read_csv(final_output)
    if len(output_df) == len(input_df):
        logger.info(f"[{dataset_label}] ✅ Verifica righe OK: {len(output_df)} (input == output)")
    else:
        logger.warning(f"[{dataset_label}] ⚠️  Mismatch righe: input={len(input_df)}, "
                      f"output={len(output_df)}")

    # --- STEP 3.5: PULIZIA FILE INTERMEDI ---
    # Dopo concatenazione riuscita, elimina le parti per ripulire il progetto
    cleanup_intermediate_files(part_input_files, part_output_files, dataset_label)

    # --- STEP 4: CLEANUP PASS (ri-imputa righe incomplete) ---
    # Usa la prima chiave disponibile per il cleanup (singolo thread)
    cleanup_key = API_KEYS[0]
    for pass_num in range(1, MAX_CLEANUP_PASSES + 1):
        logger.info(f"[{dataset_label}] 🧹 Cleanup pass {pass_num}/{MAX_CLEANUP_PASSES}")
        still_broken = cleanup_dataset(final_output, cleanup_key, dataset_label)
        if still_broken == 0:
            logger.info(f"[{dataset_label}] ✅ Tutte le righe sono state imputate correttamente!")
            break
        else:
            logger.warning(f"[{dataset_label}] Ancora {still_broken} righe incomplete dopo "
                         f"pass {pass_num}")

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("DLLM Mercury Imputation Pipeline v3")
    logger.info("=" * 60)
    logger.info(f"Chiavi API disponibili: {len(API_KEYS)}")
    logger.info(f"Worker paralleli per dataset: {MAX_WORKERS}")
    logger.info(f"Parti per dataset: {NUM_SPLITS}")
    logger.info(f"Force reprocess: {FORCE_REPROCESS}")
    logger.info(f"Cleanup passes: {MAX_CLEANUP_PASSES}")
    logger.info(f"Dataset da processare: {len(DATASETS)}")
    logger.info(f"Input dir:  {DATA_INPUT_DIR}")
    logger.info(f"Output dir: {DATA_OUTPUT_DIR}")
    logger.info("")

    # Validazione chiavi API
    placeholder_keys = [k for k in API_KEYS if k.startswith("INSERISCI")]
    if placeholder_keys:
        logger.error(f"⚠️  Hai {len(placeholder_keys)} chiavi API placeholder! "
                    f"Sostituiscile nel file prima di eseguire.")
        logger.error(f"   Chiavi da sostituire: {placeholder_keys}")
        sys.exit(1)

    # --- CARICA FEW-SHOT POOL ---
    _load_few_shot_pool()

    # --- TEST CHIAVI API ---
    working_key_indices = test_api_keys(API_KEYS)
    if not working_key_indices:
        logger.error("❌ Nessuna chiave API funzionante. Uscita.")
        sys.exit(1)
    elif len(working_key_indices) < len(API_KEYS):
        failed_indices = set(range(len(API_KEYS))) - set(working_key_indices)
        logger.warning(f"⚠️  {len(failed_indices)} chiavi non disponibili: "
                      f"{[i+1 for i in sorted(failed_indices)]}")
        logger.info(f"Procedo con {len(working_key_indices)} chiavi funzionanti.")
    else:
        logger.info("✅ Tutte le chiavi API sono funzionanti!")

    results = {}
    start_time = time.time()

    for mechanism, percentage in DATASETS:
        dataset_label = f"{mechanism}_{percentage}"
        dataset_start = time.time()

        try:
            success = process_dataset(mechanism, percentage)
        except Exception as e:
            logger.error(f"[{dataset_label}] Errore critico non gestito: {e}")
            logger.debug(traceback.format_exc())
            success = False

        elapsed = time.time() - dataset_start

        results[dataset_label] = success
        status = "✅ OK" if success else "❌ ERRORE"
        logger.info(f"[{dataset_label}] {status} (tempo: {elapsed/60:.1f} min)")

    # Riepilogo finale
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info("RIEPILOGO FINALE")
    logger.info(f"{'='*60}")
    for dataset_label, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"  {status} {dataset_label}")

    succeeded = sum(1 for s in results.values() if s)
    logger.info(f"\nCompletati: {succeeded}/{len(results)}")
    logger.info(f"Tempo totale: {total_time/60:.1f} min")

    if succeeded < len(results):
        logger.info("\n💡 Per riprovare i dataset falliti, ri-esegui lo script.")
        logger.info("   I dataset già completati verranno saltati automaticamente.")

    # =========================================================================
    # CLEANUP GLOBALE FINALE
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("🧹 CLEANUP GLOBALE FINALE")
    logger.info(f"{'='*60}")
    logger.info("Scansione di tutti i dataset completati per righe ancora incomplete...")

    total_still_broken = 0
    for mechanism, percentage in DATASETS:
        dataset_label = f"{mechanism}_{percentage}"
        final_csv = DATA_OUTPUT_DIR / f"heloc_DLLM_discriminative_train_{dataset_label}.csv"

        if not final_csv.exists():
            logger.info(f"  [{dataset_label}] File non esiste, skip.")
            continue

        # Conta righe problematiche
        df_check = pd.read_csv(final_csv, dtype=str, keep_default_na=False)
        broken_rows = 0
        broken_cells = 0
        for idx in range(len(df_check)):
            row = df_check.iloc[idx].to_dict()
            missing = get_missing_fields(row)
            if missing:
                broken_rows += 1
                broken_cells += len(missing)

        if broken_rows == 0:
            logger.info(f"  [{dataset_label}] ✅ Perfetto, nessuna riga da correggere")
        else:
            logger.warning(f"  [{dataset_label}] ⚠️  {broken_rows} righe con {broken_cells} celle da imputare")
            total_still_broken += broken_rows

            # Lancia cleanup
            cleanup_key = API_KEYS[0]
            for pass_num in range(1, MAX_CLEANUP_PASSES + 1):
                logger.info(f"  [{dataset_label}] 🧹 Cleanup globale pass {pass_num}/{MAX_CLEANUP_PASSES}")
                still_broken = cleanup_dataset(final_csv, cleanup_key, dataset_label)
                if still_broken == 0:
                    logger.info(f"  [{dataset_label}] ✅ Tutte le righe sistemate!")
                    break
                else:
                    logger.warning(f"  [{dataset_label}] Ancora {still_broken} righe dopo pass {pass_num}")

    # Riepilogo finale post-cleanup (rilegge i CSV solo se non erano già stati analizzati sopra)
    logger.info(f"\n{'='*60}")
    logger.info("RIEPILOGO POST-CLEANUP")
    logger.info(f"{'='*60}")
    for mechanism, percentage in DATASETS:
        dataset_label = f"{mechanism}_{percentage}"
        final_csv = DATA_OUTPUT_DIR / f"heloc_DLLM_discriminative_train_{dataset_label}.csv"
        if not final_csv.exists():
            logger.info(f"  ❌ {dataset_label}: file non esiste")
            continue
        df_final = pd.read_csv(final_csv, dtype=str, keep_default_na=False)
        broken = sum(
            1 for idx in range(len(df_final))
            if get_missing_fields(df_final.iloc[idx].to_dict())
        )
        status = "✅" if broken == 0 else "⚠️ "
        msg = "tutto pulito" if broken == 0 else f"{broken} ancora incomplete"
        log_fn = logger.info if broken == 0 else logger.warning
        log_fn(f"  {status} {dataset_label}: {len(df_final)} righe, {msg}")

    # --- PULIZIA DIRECTORY DISCRIMINATIVEPARTS ---
    # Se la directory è vuota (tutti i file intermedi sono stati eliminati), rimuovila
    if PARTS_OUTPUT_DIR.exists():
        remaining_files = list(PARTS_OUTPUT_DIR.iterdir())
        if not remaining_files:
            try:
                PARTS_OUTPUT_DIR.rmdir()
                logger.info(f"🗑️  Directory '{PARTS_OUTPUT_DIR.name}' vuota eliminata.")
            except OSError as e:
                logger.warning(f"Impossibile eliminare directory '{PARTS_OUTPUT_DIR.name}': {e}")
        else:
            logger.info(f"📁 Directory '{PARTS_OUTPUT_DIR.name}' contiene ancora "
                       f"{len(remaining_files)} file (parti non completate o in errore).")

    total_elapsed = time.time() - start_time
    logger.info(f"\nTempo totale (con cleanup): {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()

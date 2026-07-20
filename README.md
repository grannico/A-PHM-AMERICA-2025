# PHM North America 2025 — Manutenzione Predittiva di Motori Aeronautici


Progetto universitario sviluppato per la competizione **PHM Society North America 2025**, focalizzata sul *Prognostics and Health Management* (PHM) di motori turbofan commerciali. L'obiettivo è predire il **Remaining Useful Life (RUL)** — espresso in cicli di volo — fino a tre eventi di manutenzione distinti.

---

## Indice

1. [Contesto della Competizione](#1-contesto-della-competizione)
2. [Struttura del Progetto](#2-struttura-del-progetto)
3. [Tecnologie Utilizzate](#3-tecnologie-utilizzate)
4. [Il Dataset](#4-il-dataset)
5. [Pipeline Completa](#5-pipeline-completa)
   - [Step 1 — Pulizia Chirurgica (SurgicalCleaning)](#step-1--pulizia-chirurgica-surgicalcleaning)
   - [Step 2 — Feature Engineering Fisico (Ciclo di Brayton)](#step-2--feature-engineering-fisico-ciclo-di-brayton)
   - [Step 3 — Preparazione Separata per Evento](#step-3--preparazione-separata-per-evento)
   - [Step 4 — Modello GradientBoosting + GroupKFold](#step-4--modello-gradientboosting--groupkfold)
   - [Step 5 — Ottimizzazione OOF dei Margini (Nelder-Mead)](#step-5--ottimizzazione-oof-dei-margini-nelder-mead)
   - [Step 6 — Catena Domino: Stacking Termodinamico](#step-6--catena-domino-stacking-termodinamico)
6. [Metrica Ufficiale](#6-metrica-ufficiale)
7. [Evoluzione dello Score](#7-evoluzione-dello-score)
8. [Refactoring del Codice](#8-refactoring-del-codice)
9. [Come Riprodurre i Risultati](#9-come-riprodurre-i-risultati)
10. [Autori](#10-autori)

---

## 1. Contesto della Competizione

La **PHM Society North America 2025** ha proposto una sfida di predizione multi-target su motori turbofan commerciali. Ogni team riceve dati telemetrici storici di 4 motori e deve stimare, per un set di motori mai visti, quanti cicli di volo mancano a ciascuno dei tre eventi di manutenzione seguenti:

| Sigla | Target | Componente fisico | Tipo di evento |
|---|---|---|---|
| `WW` | `Cycles_to_WW` | Compressore (sezione fredda) | **Periodico** — Water Wash anti-fouling |
| `HPC_SV` | `Cycles_to_HPC_SV` | Compressore Alta Pressione | **Guasto** — degrado pale statoriche |
| `HPT_SV` | `Cycles_to_HPT_SV` | Turbina Alta Pressione | **Guasto** — degrado pale statoriche |

La metrica è **asimmetrica**: prevedere in ritardo (ottimismo eccessivo) pesa **il doppio** rispetto a prevedere in anticipo (conservativismo).

---

## 2. Struttura del Progetto

```
A-PHM-AMERICA-2025/
│
├── 📓 phm_refactor_post_fix.ipynb       ← Notebook FINALE (da eseguire per riprodurre)
├── 📓 phm_rul_prediction.ipynb          ← Notebook iterativo originale (storia sviluppo)
├── 📓 SurgicalCleaning.ipynb            ← Step 1: pulizia IQR del dataset raw
│
├── 📊 feature_importance_modello_finale.png  ← Grafico importanza feature
│
├── data/
│   ├── train/                           ← Dataset grezzo di training
│   │   └── training_data.csv            ← ~18.8 MB, 4 motori (ESN 101–104)
│   ├── val/                             ← Motori di validazione (un CSV per motore)
│   └── test/                            ← Motori di test/gara (un CSV per motore)
│
├── data_elaborated/
│   └── train/
│       ├── train_cleaned.csv            ← Output Step 1 (TRAIN_PRIMARY)
│       └── train_with_physics_residuals.csv  ← Fallback con residui fisici
│
├── risultati/
│   ├── submission_val_Domino.csv        ← Predizioni validation — catena Domino
│   ├── submission_test_residuals.csv    ← Predizioni test (versioni precedenti)
│   └── submission_residuals_validation.csv
│
└── archive/
    ├── notebooks/                       ← Versioni precedenti e sperimentali
    │   ├── provaNic.ipynb               ← Analisi esplorativa (EDA, motore 104)
    │   ├── prova2Nic.ipynb              ← Primo feature engineering fisico
    │   ├── AddestramentoResidui.ipynb   ← Baseline con scoring ufficiale
    │   ├── CreaResiduo.ipynb            ← Residui statistici (approccio alternativo)
    │   ├── GraficiMotori.ipynb          ← Visualizzazioni EDA
    │   └── ...
    └── immagini/                        ← Grafici e analisi intermedie
```

---

## 3. Tecnologie Utilizzate

### Linguaggio
- **Python 3.10+**

### Librerie Core

| Libreria | Versione consigliata | Utilizzo nel progetto |
|---|---|---|
| `pandas` | ≥ 2.0 | Manipolazione dati, aggregazione, rolling windows |
| `numpy` | ≥ 1.24 | Calcoli numerici, parametri fisici Brayton |
| `scikit-learn` | ≥ 1.3 | `GradientBoostingRegressor`, `GroupKFold`, `SimpleImputer` |
| `scipy` | ≥ 1.10 | `scipy.optimize.minimize` (Nelder-Mead) |
| `matplotlib` | ≥ 3.7 | Visualizzazione feature importance |

### Ambiente di Sviluppo
- **Jupyter Notebook / JupyterLab** — sviluppo iterativo della pipeline
- **Git** — controllo versione

### Installazione dipendenze

```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

oppure con un ambiente virtuale:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn scipy matplotlib jupyter
```

---

## 4. Il Dataset

### Struttura
- **4 motori** nel training set, identificati dall'**Engine Serial Number (ESN)**: 101, 102, 103, 104
- Ogni motore ha una storia completa di migliaia di cicli di volo
- Ogni **ciclo di volo** (ESN + `Cycles_Since_New`) è rappresentato da più righe, dette **snapshot**, corrispondenti a diverse fasi di volo (decollo, crociera, atterraggio, ecc.)

### Variabili Principali

| Categoria | Variabile | Descrizione |
|---|---|---|
| Identificatori | `ESN`, `Cycles_Since_New`, `Snapshot` | Motore, ciclo, fase di volo |
| Condizioni di volo | `Sensed_Altitude`, `Sensed_Mach`, `Sensed_TRA` | Quota [ft], Mach, angolo manetta |
| Temperature | `Sensed_T25`, `Sensed_T3`, `Sensed_T45`, `Sensed_T5` | Temperatura stadi motore [°R] |
| Pressioni | `Sensed_Pt2`, `Sensed_Ps3`, `Sensed_P25` | Pressioni totale e statica [psia] |
| Velocità | `Sensed_Core_Speed` (N2), `Sensed_Fan_Speed` (N1) | Velocità rotori [rpm] |
| Portata | `Sensed_WFuel` | Portata massica carburante |
| Contatori | `Cumulative_WWs`, `Cumulative_HPC_SVs`, `Cumulative_HPT_SVs` | Manutenzioni effettuate |
| **Target** | `Cycles_to_WW`, `Cycles_to_HPC_SV`, `Cycles_to_HPT_SV` | **RUL da predire** |

### Il Motore 104 — Caso Anomalo

> ⚠️ Il motore 104 presenta `std(residuo T45) = 45.97` vs `25.94` del motore 101.

Con soli 4 motori, la Group K-Fold lascia sempre un motore interamente fuori dal training. Il motore 104, il più rumoroso e anomalo, è sistematicamente il fold più difficile (gap Train/Val > 14 punti) ed è stato usato come **banco di test critico** per tutte le decisioni architetturali.

---

## 5. Pipeline Completa

```
training_data.csv (raw)
        │
        ▼
┌─────────────────────────────┐
│  STEP 1: SurgicalCleaning   │  ← SurgicalCleaning.ipynb
│  IQR clipping per (ESN, Snapshot)
└───────────────┬─────────────┘
                │ train_cleaned.csv
                ▼
┌─────────────────────────────┐
│  STEP 2: Feature Engineering│  ← add_physics_features()
│  Ciclo di Brayton (θ, δ)    │    Parametri corretti + eff. isentropica
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────┐
│  STEP 3: Prep. per Evento   │  ← prepare_mechanical_data()
│  Filtro regime + feature    │    prepare_wash_data()
│  temporali (_smooth, _trend)│
└───────────────┬─────────────┘
                │
                ▼
┌─────────────────────────────┐
│  STEP 4: Training           │  ← GradientBoostingRegressor
│  GroupKFold(4) per ESN      │    Huber loss, modelli separati
│  + SimpleImputer            │
└───────────────┬─────────────┘
                │ OOF predictions
                ▼
┌─────────────────────────────┐
│  STEP 5: OOF Margin Optim.  │  ← optimize_margins_oof()
│  Nelder-Mead su metrica PHM │    pred_finale = pred * a + b
│  esatta (asimmetrica)       │
└───────────────┬─────────────┘
                │ modelli + (a*, b*)
                ▼
┌─────────────────────────────┐
│  STEP 6: Catena Domino      │  ← train_domino_chain()
│  WW → HPC → HPT             │    OOF WW come feature di HPC
│  (stacking termodinamico)   │    OOF WW+HPC come feature di HPT
└───────────────┬─────────────┘
                │
                ▼
        submission.csv
```

---

### Step 1 — Pulizia Chirurgica (SurgicalCleaning)

**File:** `SurgicalCleaning.ipynb`  
**Output:** `data_elaborated/train/train_cleaned.csv`

Il dataset raw contiene spike di misura e outlier legati ai transitori di fase. Invece di eliminare righe (che ridurrebbe il dataset esiguo), si applica il **clipping IQR** raggruppando per `(ESN, Snapshot)`.

```python
def surgical_clean(df, is_training=True):
    # 1. Rimozione duplicati su (ESN, Cycles_Since_New, Snapshot, *sensori)
    # 2. Filtro fisico: Sensed_Altitude >= 0
    # 3. Bias training: Cycles_Since_New < 20.000
    for (esn, snap), group in df.groupby(['ESN', 'Snapshot']):
        for sensore in sensori_nomi:
            Q1 = group[sensore].quantile(0.25)
            Q3 = group[sensore].quantile(0.75)
            IQR = Q3 - Q1
            group[sensore] = group[sensore].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
```

| Parametro | Valore | Motivazione |
|---|---|---|
| Moltiplicatore IQR | `1.5` | Soglia di Tukey standard |
| Raggruppamento | `(ESN, Snapshot)` | Range accettabile varia per motore **e** per fase di volo |
| Limite cicli training | `< 20.000` | Riduce rumore nella coda a lunga vita |
| Tipo pulizia | Clipping, non drop | Preserva tutti i campioni |

> 💡 La temperatura T45 in decollo è 300–400 °F più alta che in crociera. Raggruppare per fase di volo evita di eliminare valori fisicamente plausibili nelle fasi ad alta sollecitazione.

---

### Step 2 — Feature Engineering Fisico (Ciclo di Brayton)

**Funzione:** `add_physics_features()` in `phm_refactor_post_fix.ipynb`

I segnali raw variano fortemente con le condizioni operative (quota, Mach). Per isolare il **segnale di degrado reale**, i parametri vengono normalizzati secondo le leggi termodinamiche del **Ciclo di Brayton**, il ciclo ideale dei motori turbofan.

```python
# Costanti Standard Sea Level (unità imperiali)
STD_TEMP_R = 518.67   # [Rankine]
STD_PRES   = 14.696   # [psia]
GAMMA_AIR  = 1.4      # [-]

# Parametri adimensionali di correzione
theta (θ) = Sensed_T25  / STD_TEMP_R
delta (δ) = Sensed_Pt2  / STD_PRES

# Feature corrette generate
Phy_Core_Speed_Corr = N2   / sqrt(θ)          # velocità rotore corretta
Phy_WFuel_Corr      = WFuel / (δ * sqrt(θ))   # portata carburante corretta
Phy_T45_Corr        = T45  / θ                # temperatura turbina corretta

# Efficienza isentropica compressore HPC
pr    = Sensed_Ps3 / Sensed_P25               # rapporto di compressione
k     = (GAMMA_AIR - 1) / GAMMA_AIR           # = 0.2857
T_iso = Sensed_T25 * (pr ** k)                # T ideale uscita compressore
Phy_Compressor_Eff = (T_iso - T_in) / (T_out - T_in)

# Indice termico globale
Phy_Heat_Index = Sensed_T45 / Sensed_Ps3
```

| Feature generata | Formula | Significato fisico |
|---|---|---|
| `Phy_Core_Speed_Corr` | N2 / √θ | Velocità rotore indipendente dalla temperatura |
| `Phy_WFuel_Corr` | WFuel / (δ·√θ) | Efficienza consumo carburante normalizzata |
| `Phy_T45_Corr` | T45 / θ | Temperatura turbina HPT normalizzata |
| `Phy_Compressor_Eff` | (T_iso − T_in) / (T_out − T_in) | **Efficienza isentropica HPC** — principale indicatore di degrado |
| `Phy_Heat_Index` | T45 / Ps3 | Indice termico globale del ciclo |



---

### Step 3 — Preparazione Separata per Evento

**Funzioni:** `prepare_mechanical_data()` e `prepare_wash_data()`

WW e guasti meccanici hanno dinamiche fisiche diverse → due pipeline di preparazione che filtrano **fasi operative distinte** e costruiscono **feature set specializzati**.

#### `prepare_mechanical_data()` — per HPC e HPT

Filtra il **regime di crociera stabile** (`Sensed_Altitude > 20.000 ft`), dove il segnale di degrado è meno contaminato da transitori.

```python
MECH_PHY_COLS = ['Phy_T45_Corr', 'Phy_Compressor_Eff', 'Phy_Heat_Index', 'Phy_Core_Speed_Corr']
MECH_RAW_COLS = ['Sensed_Ps3', 'Sensed_T3']

# Feature temporali generate per ogni colonna:
# col_mean   → media snapshot per ciclo
# col_smooth → rolling mean  (finestra MECH_ROLL_WINDOW = 10 cicli)
# col_trend  → differenza a  MECH_TREND_PERIODS = 5 passi
# col_std    → rolling std   (finestra 10 cicli)
# Totale: ~25 feature
```

#### `prepare_wash_data()` — per WW

Filtra il **regime di alta potenza** (`Sensed_Core_Speed > 8.000 rpm`), fase in cui i depositi si formano più rapidamente.

```python
WASH_PHY_COLS = ['Phy_Compressor_Eff', 'Phy_Heat_Index', 'Phy_WFuel_Corr']
WASH_RAW_COLS = ['Sensed_WFuel', 'Sensed_T45']

# Feature temporali generate:
# col_mean/_max → aggregazione (media + massimo per ciclo)
# col_lag1..3   → WASH_N_LAGS = 3 valori passati
# col_smooth    → rolling mean (finestra WASH_ROLL_WINDOW = 5)
# col_diff      → differenza primo ordine
# col_std       → rolling std

# Feature speciale: Cycles_Since_Last_Wash
#   = cicli dall'ultimo lavaggio (da Cumulative_WWs via diff + cumcount)
# Totale: ~71 feature
```

| Aspetto | `prepare_mechanical_data` | `prepare_wash_data` |
|---|---|---|
| Filtro regime | Altitude > 20.000 ft | Core_Speed > 8.000 rpm |
| Aggregazione | `mean` per ciclo | `mean + max` per ciclo |
| Feature temporali | `_smooth`, `_trend`, `_std` | `_lag1..3`, `_smooth`, `_diff`, `_std` |
| Feature speciale | — | `Cycles_Since_Last_Wash` |
| Totale feature | ~25 | ~71 |

---

### Step 4 — Modello GradientBoosting + GroupKFold

**Modello:** `sklearn.ensemble.GradientBoostingRegressor` con **Huber loss**  
La Huber loss combina penalità quadratica (errori piccoli) con penalità lineare (errori grandi), rendendosi robusta agli outlier del motore 104.

```python
MODEL_PARAMS = {
    "WW":  {"loss": "huber", "n_estimators": 300, "max_depth": 6,
            "learning_rate": 0.05, "subsample": 1.0, "random_state": 42},
    "HPC": {"loss": "huber", "n_estimators": 500, "max_depth": 4,
            "learning_rate": 0.03, "subsample": 0.7, "random_state": 42},
    "HPT": {"loss": "huber", "n_estimators": 500, "max_depth": 4,
            "learning_rate": 0.03, "subsample": 0.7, "random_state": 42},
}
```

**Validazione:** `GroupKFold(n_splits=4)` con gruppi per ESN — il motore di validazione non appare mai nel training del fold corrispondente, simulando il deployment su motori completamente nuovi.

| Fold | Training | Validazione |
|---|---|---|
| 1 | ESN 102, 103, 104 | ESN 101 |
| 2 | ESN 101, 103, 104 | ESN 102 |
| 3 | ESN 101, 102, 104 | ESN 103 |
| 4 | ESN 101, 102, 103 | **ESN 104** ← fold più difficile |

I valori mancanti sono gestiti con `SimpleImputer(strategy='mean')`, serializzato e riapplicato identicamente in inferenza.

---

### Step 5 — Ottimizzazione OOF dei Margini (Nelder-Mead)

**Funzione:** `optimize_margins_oof()`

La metrica penalizza doppiamente le previsioni tardive: il modello ottimale deve sistematicamente **anticipare** le manutenzioni. Il bias ottimale per componente viene trovato senza data leakage usando le **predizioni Out-of-Fold (OOF)**.

```python
def optimize_margins_oof(X_df, y_series, groups, model_params, ctype, max_train_val):
    # 1. GroupKFold → predizioni OOF "cieche"
    oof_preds = np.zeros(len(X_df))
    for train_idx, val_idx in gkf.split(X, y, groups):
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = model.predict(X[val_idx])

    # 2. Funzione obiettivo: metrica PHM esatta
    def objective(params):
        a, b = params
        y_adj = np.maximum(0, oof_preds * a + b)
        return get_exact_competition_score(y_true, y_adj, ctype, max_train_val)

    # 3. Nelder-Mead (ottimizzazione simplex, senza gradiente)
    res = minimize(objective, x0=MARGIN_INIT, method="Nelder-Mead")
    best_a, best_b = res.x
    # pred_finale = max(0, pred_raw * best_a + best_b)

    # 4. Riaddestramento finale su TUTTI i dati
    # 5. Restituzione di oof_preds "puri" per la catena Domino
```

**Risultati dell'ottimizzazione:**

| Componente | a\* | b\* | Score OOF raw | Score OOF ottimizzato | Δ |
|---|---|---|---|---|---|
| WW | 0.1396 | +163.65 | 50.51 | 19.79 | −60.8% |
| HPC | 0.4657 | +256.94 | 72.19 | 42.48 | −41.1% |
| HPT | 0.4670 | −7.84 | 88.63 | 29.10 | −67.2% |


---

### Step 6 — Catena Domino: Stacking Termodinamico

**Funzione:** `train_domino_chain()`

I tre eventi non sono indipendenti: il **Water Wash** rimuove i depositi dal compressore, influenzando il degrado dell'**HPC Stator Vane**; il comportamento del compressore determina le condizioni di ingresso alla turbina, influenzando l'**HPT Stator Vane**.

La catena Domino traduce queste correlazioni fisiche in **feature informative per i modelli downstream**.

```
WW  ──(OOF pure)──▶  HPC  ──(OOF pure)──▶  HPT
 │                    │                      │
 └── pred*a+b         └── pred*a+b           └── pred*a+b
   (submission)         (submission)           (submission)
```

```python
tasks = [
    {"target": "Cycles_to_WW",     "data": df_wash, "type": "WW"},   # 1° — no upstream
    {"target": "Cycles_to_HPC_SV", "data": df_mech, "type": "HPC"},  # 2° — usa OOF WW
    {"target": "Cycles_to_HPT_SV", "data": df_mech, "type": "HPT"},  # 3° — usa OOF WW+HPC
]

# In inferenza (generate_submission_file):
for ctype in ["WW", "HPC", "HPT"]:      # ordine tassativo
    if ctype == "HPC":
        df["Pred_WW"]  = pure_preds["WW"]
    elif ctype == "HPT":
        df["Pred_WW"]  = pure_preds["WW"]
        df["Pred_HPC"] = pure_preds["HPC"]
    pred_raw  = model.predict(X_clean)[-1]
    pred_safe = max(0, pred_raw * best_a + best_b)
    pure_preds[ctype] = pred_raw      # per i modelli successivi
    row_pred[ctype]   = pred_safe     # per la submission
```

| Componente | Feature Domino aggiuntive | Totale feature | Score OOF ottimizzato |
|---|---|---|---|
| WW | — (primo della catena) | 71 | 19.79 |
| HPC | `Pred_WW` (OOF puri) | 72 | 42.48 |
| HPT | `Pred_WW` + `Pred_HPC` (OOF puri) | 73 | 29.10 |

---

## 6. Metrica Ufficiale

```
Score = mean( w(ŷ, y) · (ŷ − y)² · β )

w(ŷ, y) = SCORE_W_OVER / (1 + α·y)   se ŷ ≥ y   [sovrastima — penalità doppia]
w(ŷ, y) = SCORE_W_UNDER / (1 + α·y)  se ŷ < y   [sottostima]

α               = SCORE_ALPHA     = 0.01
β (WW)          = SCORE_BETA_WW   / max(y_train) = 1 / max(y)
β (HPC, HPT)    = SCORE_BETA_MECH / max(y_train) = 2 / max(y)
SCORE_W_OVER    = 2.0   (peso sovrastima)
SCORE_W_UNDER   = 1.0   (peso sottostima)
```

**Implementata in:** `get_exact_competition_score()` — usata come funzione obiettivo diretta in Nelder-Mead.

---

## 7. Evoluzione dello Score

| Iterazione | Notebook / Step | Innovazione introdotta | Score |
|---|---|---|---|
| 1 | `phm_rul_prediction` v1 | Feature fisiche Brayton; 3 modelli separati | 219 |
| 2 | `phm_rul_prediction` v2 | Bug fix normalizzazione θ/δ; armonizzazione colonne | 190.4 |
| 3 | `phm_rul_prediction` v3 | Feature `_trend` (differenza a 5 passi) | 167.9 |
| 4 | `phm_rul_prediction` v4 | Feature `_std` rolling std — 25 feat mech / 71 WW | 140.0 |
| 5 | `phm_rul_prediction` v5 | **OOF Margin Optimization** con Nelder-Mead | 78.26 |
| 6 | `phm_rul_prediction` v6 | **Effetto Domino** WW→HPC→HPT (stacking) | **71 ✓** |
| 7 | `phm_refactor_post_fix` | Refactoring completo, comportamento invariato | 71 |

```
219 → 190.4 → 167.9 → 140.0 → 78.26 → 71
 ▲       ▲       ▲       ▲       ▲      ▲
Bug fix Trend  Volatil. OOF   Domino
        feat.  feat.    Opt.
```

---

## 9. Come Riprodurre i Risultati

### Prerequisiti

```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

### Ordine di esecuzione

**Step 1 — Pulizia dati** (eseguire una sola volta):

Aprire ed eseguire `SurgicalCleaning.ipynb` (Restart & Run All).  
Output: `data_elaborated/train/train_cleaned.csv`

**Step 2 — Training e generazione submission** (notebook principale):

Aprire `phm_refactor_post_fix.ipynb` e fare **Restart & Run All**.

Il notebook:
1. Carica `train_cleaned.csv`
2. Armonizza i nomi delle colonne (`harmonize_columns`)
3. Aggiunge le feature fisiche Brayton (`add_physics_features`)
4. Prepara i dataset per evento (`prepare_mechanical_data`, `prepare_wash_data`)
5. Esegue la catena Domino con OOF optimization (`train_domino_chain`)
6. Genera le submission su `data/val/` e `data/test/`

**Struttura dati richiesta:**
```
data/
  train/training_data.csv    ← dataset grezzo
  val/                       ← un CSV per motore (validation)
  test/                      ← un CSV per motore (test/gara)
```

**Output attesi:**
```
risultati/submission_val_Domino.csv    ← predizioni validation
risultati/submission_final.csv         ← predizioni test (submission ufficiale)
```

> ℹ️ Se `train_cleaned.csv` non è disponibile, il notebook usa automaticamente `train_with_physics_residuals.csv` come fallback (`TRAIN_FALLBACK`).

---

## 10. Autori

**Balloni Niccolò** **Concetti Francesco** **Giannetti Lorenzo**  
Progetto universitario per il corso di Manutenzione Preventiva per la Robotica e l'Automazione Intelligente— A.A. 2025/2026  
Competizione: [PHM Society North America 2025](https://phmsociety.org/)

---

*Score finale: **75.25** *
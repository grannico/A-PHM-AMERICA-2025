"""
Script diagnostico: quantifica il disallineamento tra df_wash e df_mech
nel merging delle predizioni OOF (catena Domino).
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ---- Riproduce la logica del notebook per caricare e preparare i dati ----
# Importiamo le costanti e le funzioni direttamente dal notebook convertendolo
# in un modulo temporaneo.

# Per brevità, ri-definiamo solo le costanti necessarie e le funzioni di preparazione
# copiandole dal notebook.

RANDOM_STATE = 42
DATA_ELAB_TRAIN_DIR = os.path.join("data_elaborated", "train")
TRAIN_PRIMARY   = os.path.join(DATA_ELAB_TRAIN_DIR, "train_cleaned.csv")
TRAIN_FALLBACK  = os.path.join(DATA_ELAB_TRAIN_DIR, "train_with_physics_residuals.csv")

TARGETS = ["Cycles_to_HPC_SV", "Cycles_to_HPT_SV", "Cycles_to_WW"]

STD_TEMP_R = 518.67
STD_PRES   = 14.696
GAMMA_AIR  = 1.4

ALT_THRESHOLD_MECH   = 20000
SPEED_THRESHOLD_WASH = 8000

MECH_PHY_COLS = ["Phy_T45_Corr", "Phy_Compressor_Eff", "Phy_Heat_Index", "Phy_Core_Speed_Corr"]
MECH_RAW_COLS = ["Sensed_Ps3", "Sensed_T3"]
MECH_ROLL_WINDOW   = 10
MECH_TREND_PERIODS = 5

WASH_PHY_COLS = ["Phy_Compressor_Eff", "Phy_Heat_Index", "Phy_WFuel_Corr"]
WASH_RAW_COLS = ["Sensed_WFuel", "Sensed_T45"]
WASH_ROLL_WINDOW = 5
WASH_N_LAGS      = 3

SENSOR_RENAME_MAP = {
    "Cycles": "Cycles_Since_New", "Cycle": "Cycles_Since_New",
    "Altitude": "Sensed_Altitude", "Mach": "Sensed_Mach",
    "TRA": "Sensed_TRA", "T2": "Sensed_T2", "T24": "Sensed_T24",
    "T25": "Sensed_T25", "Pt2": "Sensed_Pt2",
    "W": "Sensed_WFuel", "WFuel": "Sensed_WFuel",
    "Core_Speed": "Sensed_Core_Speed", "N2": "Sensed_Core_Speed",
    "Fan_Speed": "Sensed_Fan_Speed", "N1": "Sensed_Fan_Speed",
    "T30": "Sensed_T3", "T3": "Sensed_T3",
    "T48": "Sensed_T45", "T45": "Sensed_T45", "T50": "Sensed_T5",
    "P15": "Sensed_P15", "P2": "Sensed_P2", "P21": "Sensed_P21",
    "P24": "Sensed_P24", "P25": "Sensed_P25",
    "Ps30": "Sensed_Ps3", "Ps3": "Sensed_Ps3", "P40": "Sensed_P40",
    "P50": "Sensed_P50", "HPC_SV": "Cycles_to_HPC_SV",
    "HPT_SV": "Cycles_to_HPT_SV", "WW": "Cycles_to_WW",
}
ID_COLS = ["ESN", "Cycles_Since_New", "Snapshot", "File_ID", "file"]


def harmonize_columns(df):
    df = df.rename(columns=SENSOR_RENAME_MAP)
    new_cols = {}
    for col in df.columns:
        if col not in ID_COLS:
            if not col.startswith("Sensed_") and not col.startswith("Phy_") and not col.startswith("Cycles_to_"):
                new_cols[col] = f"Sensed_{col}"
    if new_cols:
        df = df.rename(columns=new_cols)
    return df


def add_physics_features(df):
    df = df.copy()
    if "Sensed_T25" in df.columns and "Sensed_Pt2" in df.columns:
        theta = df["Sensed_T25"] / STD_TEMP_R
        theta = np.maximum(theta, 0.0001)
        delta = df["Sensed_Pt2"] / STD_PRES
        theta = theta.replace(0, 1); delta = delta.replace(0, 1)
        if "Sensed_Core_Speed" in df.columns:
            df["Phy_Core_Speed_Corr"] = df["Sensed_Core_Speed"] / np.sqrt(theta)
        if "Sensed_WFuel" in df.columns:
            df["Phy_WFuel_Corr"] = df["Sensed_WFuel"] / (delta * np.sqrt(theta))
        if "Sensed_T45" in df.columns:
            df["Phy_T45_Corr"] = df["Sensed_T45"] / theta
        if "Sensed_T3" in df.columns and "Sensed_Ps3" in df.columns:
            T_in_R = df["Sensed_T25"]
            T_out_R = df["Sensed_T3"]
            P_in = df["Sensed_P25"] if "Sensed_P25" in df.columns else df["Sensed_Pt2"]
            P_out = df["Sensed_Ps3"]
            pr = P_out / P_in
            k = (GAMMA_AIR - 1) / GAMMA_AIR
            T_iso_R = T_in_R * (pr ** k)
            df["Phy_Compressor_Eff"] = (T_iso_R - T_in_R) / (T_out_R - T_in_R)
    if "Sensed_T45" in df.columns and "Sensed_Ps3" in df.columns:
        df["Phy_Heat_Index"] = df["Sensed_T45"] / df["Sensed_Ps3"]
    return df


def prepare_mechanical_data(df, is_test=False):
    df = df.copy()
    df = harmonize_columns(df)
    if "Sensed_Altitude" in df.columns:
        df = df[df["Sensed_Altitude"] > ALT_THRESHOLD_MECH].copy()
    df = add_physics_features(df)
    use_cols = [c for c in MECH_PHY_COLS + MECH_RAW_COLS if c in df.columns]
    agg_dict = {col: "mean" for col in use_cols}
    for t in TARGETS:
        if t in df.columns:
            agg_dict[t] = "first"
    if not is_test and "ESN" in df.columns:
        df_grouped = df.groupby(["ESN", "Cycles_Since_New"]).agg(agg_dict).reset_index()
        df_grouped = df_grouped.sort_values(["ESN", "Cycles_Since_New"])
        for col in use_cols:
            df_grouped[f"{col}_smooth"] = df_grouped.groupby("ESN")[col].transform(lambda x: x.rolling(window=MECH_ROLL_WINDOW, min_periods=1).mean())
            df_grouped[f"{col}_trend"] = df_grouped.groupby("ESN")[f"{col}_smooth"].diff(periods=MECH_TREND_PERIODS).fillna(0)
            df_grouped[f"{col}_std"] = df_grouped.groupby("ESN")[col].transform(lambda x: x.rolling(window=MECH_ROLL_WINDOW, min_periods=1).std()).fillna(0)
    return df_grouped.ffill().bfill().fillna(0)


def prepare_wash_data(df, is_test=False):
    df = df.copy()
    df = harmonize_columns(df)
    if "Sensed_Core_Speed" in df.columns:
        df = df[df["Sensed_Core_Speed"] > SPEED_THRESHOLD_WASH].copy()
    df = add_physics_features(df)
    use_cols = [c for c in WASH_PHY_COLS + WASH_RAW_COLS if c in df.columns]
    agg_dict = {}
    for c in use_cols:
        agg_dict[c] = ["mean", "max"]
    if "Cycles_to_WW" in df.columns:
        agg_dict["Cycles_to_WW"] = "first"
    if "Cumulative_WWs" in df.columns:
        agg_dict["Cumulative_WWs"] = "max"
    if not is_test and "ESN" in df.columns:
        df_grouped = df.groupby(["ESN", "Cycles_Since_New"]).agg(agg_dict)
    else:
        df_grouped = df.groupby("Cycles_Since_New").agg(agg_dict)
    new_cols = []
    feature_cols = []
    for c, s in df_grouped.columns:
        if c in ["Cycles_to_WW", "Cumulative_WWs"] or s == "":
            new_cols.append(c)
        else:
            name = f"{c}_{s}"
            new_cols.append(name)
            feature_cols.append(name)
    df_grouped.columns = new_cols
    df_grouped = df_grouped.reset_index()
    if not is_test and "ESN" in df.columns:
        df_grouped = df_grouped.sort_values(["ESN", "Cycles_Since_New"])
        if "Cumulative_WWs" in df_grouped.columns:
            df_grouped["WW_Change"] = df_grouped.groupby("ESN")["Cumulative_WWs"].diff().fillna(0)
            df_grouped["Wash_Session_ID"] = df_grouped.groupby("ESN")["WW_Change"].cumsum()
            df_grouped["Cycles_Since_Last_Wash"] = df_grouped.groupby(["ESN", "Wash_Session_ID"]).cumcount()
            feature_cols.append("Cycles_Since_Last_Wash")
        for col in feature_cols:
            if col == "Cycles_Since_Last_Wash":
                continue
            for i in range(1, WASH_N_LAGS + 1):
                df_grouped[f"{col}_lag{i}"] = df_grouped.groupby("ESN")[col].shift(i)
            df_grouped[f"{col}_smooth"] = df_grouped.groupby("ESN")[col].transform(lambda x: x.rolling(window=WASH_ROLL_WINDOW, min_periods=1).mean())
            df_grouped[f"{col}_diff"] = df_grouped.groupby("ESN")[col].diff()
            df_grouped[f"{col}_std"] = df_grouped.groupby("ESN")[col].transform(lambda x: x.rolling(window=WASH_ROLL_WINDOW, min_periods=1).std()).fillna(0)
    df_grouped = df_grouped.ffill().bfill().fillna(0)
    drop_cols = ["WW_Change", "Wash_Session_ID", "Cumulative_WWs"]
    return df_grouped.drop(columns=[c for c in drop_cols if c in df_grouped.columns])


# ======================= DIAGNOSI =======================
print("=" * 70)
print("DIAGNOSI DATA LEAKAGE - FALLBACK OOF NELLA CATENA DOMINO")
print("=" * 70)

# 1. Caricamento dati
if os.path.exists(TRAIN_PRIMARY):
    df_raw = pd.read_csv(TRAIN_PRIMARY)
    print(f"\nFile caricato: {TRAIN_PRIMARY}")
elif os.path.exists(TRAIN_FALLBACK):
    df_raw = pd.read_csv(TRAIN_FALLBACK)
    print(f"\nFile caricato: {TRAIN_FALLBACK}")
else:
    print("ERRORE: nessun file di training trovato!"); sys.exit(1)

df_raw = harmonize_columns(df_raw)

# 2. Preparazione dati
df_mech = prepare_mechanical_data(df_raw, is_test=False)
df_wash = prepare_wash_data(df_raw, is_test=False)

print(f"\ndf_mech: {df_mech.shape[0]} righe (cicli-motore in regime crociera)")
print(f"df_wash: {df_wash.shape[0]} righe (cicli-motore in regime alta potenza)")

# 3. Costruiamo gli indici multi-livello
mech_idx = set(zip(df_mech["ESN"], df_mech["Cycles_Since_New"]))
wash_idx = set(zip(df_wash["ESN"], df_wash["Cycles_Since_New"]))

print(f"\nChiavi uniche (ESN, Cycles_Since_New) in df_mech: {len(mech_idx)}")
print(f"Chiavi uniche (ESN, Cycles_Since_New) in df_wash: {len(wash_idx)}")
print(f"Intersezione (chiavi comuni):                      {len(mech_idx & wash_idx)}")
print(f"Solo in df_mech (non in df_wash):                  {len(mech_idx - wash_idx)}")
print(f"Solo in df_wash (non in df_mech):                  {len(wash_idx - mech_idx)}")

# 4. Simuliamo il merging come fa train_domino_chain
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingRegressor

GKF_SPLITS = 4

# --- Addestramento WW per ottenere le OOF ---
print("\n" + "=" * 70)
print("SIMULAZIONE: addestramento WW e mapping OOF su df_mech")
print("=" * 70)

target_ww = "Cycles_to_WW"
features_ww = [c for c in df_wash.columns if c not in ["ESN", target_ww, "Snapshot"] and "Cycles_to_" not in c]
X_ww = df_wash[features_ww].fillna(0)
y_ww = df_wash[target_ww]
groups_ww = df_wash["ESN"]

gkf = GroupKFold(n_splits=GKF_SPLITS)
oof_ww = np.zeros(len(X_ww))
for train_idx, val_idx in gkf.split(X_ww.values, y_ww.values, groups=groups_ww.values):
    m = GradientBoostingRegressor(loss="huber", n_estimators=50, max_depth=4,
                                   learning_rate=0.05, random_state=42)
    m.fit(X_ww.values[train_idx], y_ww.values[train_idx])
    oof_ww[val_idx] = m.predict(X_ww.values[val_idx])

# Creiamo la Series OOF con MultiIndex come fa il codice
oof_series_ww = pd.Series(
    oof_ww,
    index=pd.MultiIndex.from_arrays([df_wash["ESN"], df_wash["Cycles_Since_New"]])
)

print(f"\nOOF WW generate: {len(oof_series_ww)} valori")

# --- Proviamo a mappare su df_mech ---
df_test = df_mech.copy()
df_test = df_test.set_index(["ESN", "Cycles_Since_New"])
df_test["Pred_WW"] = oof_series_ww
df_test = df_test.reset_index()

n_total = len(df_test)
n_nan = df_test["Pred_WW"].isna().sum()
n_valid = n_total - n_nan

print(f"\nRighe totali in df_mech:  {n_total}")
print(f"Pred_WW valide (matched): {n_valid}")
print(f"Pred_WW NaN (mancanti):   {n_nan}")
print(f"Percentuale NaN:          {100*n_nan/n_total:.2f}%")

if n_nan > 0:
    print("\n" + "!" * 70)
    print("⚠️  CONFERMATO: ci sono NaN nel merge delle OOF!")
    print("    Il codice li riempie con: df['Pred_WW'].fillna(df['Cycles_to_WW'])")
    print("    Questo è DATA LEAKAGE: il target reale viene usato come feature!")
    print("!" * 70)
    
    # Dettaglio per motore
    print("\nDettaglio per motore:")
    for esn in sorted(df_test["ESN"].unique()):
        sub = df_test[df_test["ESN"] == esn]
        n_miss = sub["Pred_WW"].isna().sum()
        if n_miss > 0:
            print(f"  ESN={esn}: {n_miss}/{len(sub)} cicli NaN ({100*n_miss/len(sub):.1f}%)")
else:
    print("\n" + "=" * 70)
    print("✅ NESSUN NaN: il merge è perfetto, nessun fallback viene attivato!")
    print("    Il codice di fallback esiste ma NON viene mai eseguito.")
    print("=" * 70)

# 5. Verifica analoga per Pred_HPC -> HPT
print("\n\n" + "=" * 70)
print("VERIFICA ANALOGA: Pred_HPC per il modello HPT")
print("=" * 70)

# L'HPC è addestrato su df_mech, le OOF sono mappate su df_mech stesso
# Quindi dovrebbero matchare al 100%
print("Le OOF di HPC sono generate da df_mech e rimappate su df_mech stesso.")
print("Per costruzione, gli indici sono identici -> NaN impossibili per Pred_HPC.")
print("Il leakage potenziale riguarda SOLO Pred_WW (da df_wash a df_mech).")

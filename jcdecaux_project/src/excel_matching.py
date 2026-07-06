import pandas as pd
import os
import re
import unicodedata
from pathlib import Path
from config.config import EXCEL_PATH, SHEET_NAME, IMAGE_FOLDER, OUTPUT_DIR, VALID_IMAGE_EXTS

def norm_str(s):
    if s is None:
        return ""
    s = str(s).replace("\n", " ").strip()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r'["\']', '', s)
    s = re.sub(r'[\s_]+', ' ', s).strip()
    return s

# --- Lire Excel ---
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# --- Détecter colonne de noms ---
possible_name_cols = ["Nom_fichier_visuel", "nom_fichier_visuel", "nom_fichier", "fichier", "image_name"]
name_col = None
for c in possible_name_cols:
    if c in df.columns:
        name_col = c
        break

if name_col is None:
    for c in df.columns:
        lc = c.lower()
        if ("nom" in lc and ("fichier" in lc or "visuel" in lc)) or ("image" in lc and "nom" in lc):
            name_col = c
            break

if name_col is None:
    df["Nom_fichier_visuel"] = ""
    name_col = "Nom_fichier_visuel"
else:
    df[name_col] = df[name_col].fillna("").astype(str).str.strip()

print(f"Using filename column: {name_col}")

# --- Lister images ---
files_in_folder = [f for f in os.listdir(IMAGE_FOLDER) if os.path.splitext(f)[1].lower() in VALID_IMAGE_EXTS]
files_map = {}
for f in files_in_folder:
    full = os.path.join(IMAGE_FOLDER, f)
    base_norm = norm_str(os.path.splitext(f)[0])
    files_map.setdefault(base_norm, full)
    files_map.setdefault(norm_str(f), full)
    files_map.setdefault(f.lower(), full)

def find_image_path_from_excel_name(name: str):
    key = norm_str(name)
    if key in files_map:
        return files_map[key]
    for k, path in files_map.items():
        if key in k or k in key:
            return path
    for ext in VALID_IMAGE_EXTS:
        candidate = key + ext
        if candidate in files_map:
            return files_map[candidate]
    return None

rows, missing_rows = [], []
for idx, row in df.iterrows():
    img_name = row.get(name_col, "")
    img_path = find_image_path_from_excel_name(img_name)
    rec = row.to_dict()
    rec["image_path"] = img_path
    if img_path:
        rows.append(rec)
    else:
        missing_rows.append(rec)

df_labeled = pd.DataFrame(rows).reset_index(drop=True)
df_unmatched_rows = pd.DataFrame(missing_rows).reset_index(drop=True)

# --- Export CSV diagnostics ---
df_labeled.to_csv(os.path.join(OUTPUT_DIR, "df_labeled.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame({"path": sorted(list(set([p for p in df_labeled['image_path']])))}).to_csv(
    os.path.join(OUTPUT_DIR, "images_trouvees.csv"), index=False, encoding="utf-8-sig"
)
pd.DataFrame({"path": sorted(list(set([os.path.join(IMAGE_FOLDER, f) for f in files_in_folder])))}).to_csv(
    os.path.join(OUTPUT_DIR, "images_in_folder.csv"), index=False, encoding="utf-8-sig"
)
pd.DataFrame({"path": sorted(list(set([p for p in df_unmatched_rows.get(name_col, [])])))}).to_csv(
    os.path.join(OUTPUT_DIR, "excel_rows_without_image_name.csv"), index=False, encoding="utf-8-sig"
)
pd.DataFrame({"path": sorted(list(set([p for p in (set(os.path.join(IMAGE_FOLDER, f) for f in files_in_folder) - set(df_labeled['image_path'].tolist()))])))}).to_csv(
    os.path.join(OUTPUT_DIR, "images_non_trouvees.csv"), index=False, encoding="utf-8-sig"
)
print("Excel matching done.")

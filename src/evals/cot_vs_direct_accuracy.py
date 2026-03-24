import argparse
from pathlib import Path
import re
import pandas as pd


def infer_language_from_name(name: str) -> str:
    # try common patterns like _en, -en, en.csv, en
    m = re.search(r'[_\-]([a-z]{2,3})$', name, re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r'[_\-]([a-z]{2,3})\.', name, re.I)
    if m:
        return m.group(1).lower()
    # fallback to entire stem
    return Path(name).stem


def find_is_correct_col(df: pd.DataFrame) -> str | None:
    # prefer explicit majority column if present
    for col in df.columns:
        if col.lower() in {'majority_correct', 'majority-correct', 'majority'}:
            return col
    # fallback to is_correct / correct variants
    for col in df.columns:
        if col.lower() == 'is_correct':
            return col
    for col in df.columns:
        if 'is_correct' in col.lower():
            return col
    for col in df.columns:
        if col.lower() == 'correct':
            return col
    return None


def to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(bool).fillna(False)

    def parse_val(v):
        if pd.isna(v):
            return False
        vs = str(v).strip().lower()
        if vs in {'true', 't', '1', 'yes', 'y'}:
            return True
        if vs in {'false', 'f', '0', 'no', 'n'}:
            return False
        # fallback: treat numeric-like strings
        if re.fullmatch(r"\d+", vs):
            return int(vs) != 0
        return False

    return s.map(parse_val)


def collect_accuracies_from_path(path: Path, model_name: str) -> list[dict]:
    rows = []
    if not path.exists():
        return rows

    for f in sorted(path.iterdir()):
        if f.is_dir():
            # maybe nested structure: iterate csv/xlsx inside
            rows += collect_accuracies_from_path(f, model_name)
            continue

        if f.suffix.lower() in {'.xlsx', '.xls'}:
            try:
                sheets = pd.read_excel(f, sheet_name=None)
            except Exception:
                continue
            for sheet_name, df in sheets.items():
                col = find_is_correct_col(df)
                if col is None:
                    continue
                lang = sheet_name.lower()
                total = len(df)
                if total == 0:
                    acc = 0.0
                else:
                    acc = to_bool_series(df[col]).sum() / total
                rows.append({
                    'model': model_name,
                    'language': lang,
                    'accuracy': acc,
                    'n': total,
                    'source_file': str(f.name),
                    'sheet': sheet_name,
                })

        elif f.suffix.lower() == '.csv':
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            col = find_is_correct_col(df)
            if col is None:
                continue
            lang = infer_language_from_name(f.stem)
            total = len(df)
            acc = to_bool_series(df[col]).sum() / total if total else 0.0
            rows.append({
                'model': model_name,
                'language': lang,
                'accuracy': acc,
                'n': total,
                'source_file': str(f.name),
                'sheet': None,
            })

    return rows


def extract_lang(val) -> str:
    """Normalize language values like cot_majority_bn or direct_majority_en to just bn, en, etc."""
    if pd.isna(val):
        return ''
    s = str(val).lower()
    m = re.search(r'\b(en|zh|bn|te|sw)\b', s)
    if m:
        return m.group(1)
    m = re.search(r'[_\-]([a-z]{2,3})$', s)
    if m:
        return m.group(1)
    return s


def main(cot_dir: Path, direct_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cot_rows = []
    direct_rows = []

    # Only process final Excel files named final_data_<model>.xlsx and final_direct_<model>.xlsx
    cot_final_dir = cot_dir / 'final'
    direct_final_dir = direct_dir / 'final'

    LANG_CODES = {'en', 'bn', 'te', 'sw', 'zh'}

    def extract_from_final_excel(path: Path, model_prefix: str) -> list[dict]:
        rows = []
        if not path.exists():
            return rows
        for f in sorted(path.iterdir()):
            if not f.is_file():
                continue
            name = f.name
            if not name.lower().startswith(model_prefix):
                continue
            model = re.sub(rf'^{re.escape(model_prefix)}', '', Path(name).stem, flags=re.I)
            try:
                sheets = pd.read_excel(f, sheet_name=None)
            except Exception:
                continue
            for sheet_name, df in sheets.items():
                # First, detect any majority-like columns that include language in the name
                # e.g., 'cot_majority_en', 'direct_majority_en', 'majority_en', 'majority-correct_en'
                maj_cols = []
                for col in df.columns:
                    m = re.search(r'(?:cot_|direct_)?majority(?:[_\-]?correct)?[_\-]?([a-z]{2,3})$', col, re.I)
                    if m:
                        lang = m.group(1).lower()
                        maj_cols.append((col, lang))

                if maj_cols:
                    total = len(df)
                    for col, lang in maj_cols:
                        true_count = int(to_bool_series(df[col]).sum()) if total else 0
                        acc = true_count / total if total else 0.0
                        rows.append({
                            'model': model,
                            'language': lang,
                            'accuracy': acc,
                            'true_count': true_count,
                            'n': total,
                            'source_file': name,
                            'sheet': sheet_name,
                        })
                    continue

                # determine language: prefer sheet name if it matches known codes
                lang = sheet_name.lower()
                if lang not in LANG_CODES:
                    # check for column 'language' or 'lang'
                    for c in ['language', 'lang']:
                        if c in df.columns:
                            vals = df[c].dropna().unique()
                            if len(vals) > 0:
                                v = str(vals[0]).lower()
                                if v in LANG_CODES:
                                    lang = v
                                    break
                    else:
                        # try to infer from sheet name tokens like 'en', 'zh'
                        m = re.search(r'\b(en|bn|te|sw|zh)\b', sheet_name, re.I)
                        if m:
                            lang = m.group(1).lower()
                col = find_is_correct_col(df)
                if col is None:
                    continue
                total = len(df)
                true_count = int(to_bool_series(df[col]).sum()) if total else 0
                acc = true_count / total if total else 0.0
                rows.append({
                    'model': model,
                    'language': lang,
                    'accuracy': acc,
                    'true_count': true_count,
                    'n': total,
                    'source_file': name,
                    'sheet': sheet_name,
                })
        return rows

    cot_rows = extract_from_final_excel(cot_final_dir, 'final_data_')
    direct_rows = extract_from_final_excel(direct_final_dir, 'final_direct_')

    cot_df = pd.DataFrame(cot_rows)
    direct_df = pd.DataFrame(direct_rows)

    # Normalize language values early — handles cot_majority_bn, direct_majority_en, etc.
    if not cot_df.empty:
        cot_df['language'] = cot_df['language'].map(extract_lang)
    if not direct_df.empty:
        direct_df['language'] = direct_df['language'].map(extract_lang)

    # enforce requested language order: english, chinese, bengali, telugu, swahili
    LANG_ORDER = ['en', 'zh', 'bn', 'te', 'sw']

    def order_df_by_lang(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # normalize language codes to lower-case
        df['language'] = df['language'].astype(str).str.lower()
        # sort by a temporary categorical to preserve string dtype
        cat = pd.Categorical(df['language'], categories=LANG_ORDER, ordered=True)
        df['_lang_cat'] = cat
        df = df.sort_values(['model', '_lang_cat'])
        df.drop(columns=['_lang_cat'], inplace=True)
        # ensure language stays string
        df['language'] = df['language'].astype(str)
        return df

    cot_df = order_df_by_lang(cot_df)
    direct_df = order_df_by_lang(direct_df)

    out_xlsx = out_dir / 'final_accuracies_mgsm.xlsx'

    # ensure language column ordering and fill missing languages per model
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        if not cot_df.empty:
            cot_df.to_excel(writer, sheet_name='cot_accuracy', index=False)
        else:
            pd.DataFrame(columns=['model', 'language', 'accuracy', 'true_count', 'n', 'source_file', 'sheet']).to_excel(
                writer, sheet_name='cot_accuracy', index=False)

        if not direct_df.empty:
            direct_df.to_excel(writer, sheet_name='direct_accuracy', index=False)
        else:
            pd.DataFrame(columns=['model', 'language', 'accuracy', 'true_count', 'n', 'source_file', 'sheet']).to_excel(
                writer, sheet_name='direct_accuracy', index=False)

        # diff
        if not cot_df.empty and not direct_df.empty:
            left = cot_df.rename(columns={'accuracy': 'cot_accuracy'})[['model', 'language', 'cot_accuracy']]
            right = direct_df.rename(columns={'accuracy': 'direct_accuracy'})[['model', 'language', 'direct_accuracy']]

            merged = pd.merge(left, right, on=['model', 'language'], how='inner')
            merged['accuracy_diff'] = merged['cot_accuracy'] - merged['direct_accuracy']

            # keep only requested three columns
            diff_df = merged[['model', 'language', 'accuracy_diff']]

            # order languages and rows
            diff_df = diff_df.copy()
            diff_df['language'] = diff_df['language'].astype(str).str.lower()
            diff_df['language'] = pd.Categorical(diff_df['language'], categories=LANG_ORDER, ordered=True)
            diff_df.sort_values(['model', 'language'], inplace=True)
            # restore language to string
            diff_df['language'] = diff_df['language'].astype(str)

            diff_df.to_excel(writer, sheet_name='cot_vs_direct_diff', index=False)
        else:
            pd.DataFrame(columns=['model', 'language', 'cot_accuracy', 'direct_accuracy', 'accuracy_diff']).to_excel(
                writer, sheet_name='cot_vs_direct_diff', index=False)

    print('Wrote:', out_xlsx)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Compute CoT and Direct accuracies per model and language and export diff CSVs for MGSM.'
    )
    dataset_name = 'mmlu'
    p.add_argument('--cot-dir', type=Path, default=Path(f'results/cot_inference/{dataset_name}'))
    p.add_argument('--direct-dir', type=Path, default=Path(f'results/direct_inference/{dataset_name}'))
    p.add_argument('--out-dir', type=Path, default=Path(f'results/hypothesis_1/{dataset_name}'))
    args = p.parse_args()
    main(args.cot_dir, args.direct_dir, args.out_dir)







# import argparse
# from pathlib import Path
# import re
# import pandas as pd


# def infer_language_from_name(name: str) -> str:
#     # try common patterns like _en, -en, en.csv, en
#     m = re.search(r'[_\-]([a-z]{2,3})$', name, re.I)
#     if m:
#         return m.group(1).lower()
#     m = re.search(r'[_\-]([a-z]{2,3})\.', name, re.I)
#     if m:
#         return m.group(1).lower()
#     # fallback to entire stem
#     return Path(name).stem


# def find_is_correct_col(df: pd.DataFrame) -> str | None:
#     # prefer explicit majority column if present
#     for col in df.columns:
#         if col.lower() in {'majority_correct', 'majority-correct', 'majority'}:
#             return col
#     # fallback to is_correct / correct variants
#     for col in df.columns:
#         if col.lower() == 'is_correct':
#             return col
#     for col in df.columns:
#         if 'is_correct' in col.lower():
#             return col
#     for col in df.columns:
#         if col.lower() == 'correct':
#             return col
#     return None


# def to_bool_series(s: pd.Series) -> pd.Series:
#     if pd.api.types.is_bool_dtype(s):
#         return s.fillna(False)
#     if pd.api.types.is_numeric_dtype(s):
#         return s.astype(bool).fillna(False)

#     def parse_val(v):
#         if pd.isna(v):
#             return False
#         vs = str(v).strip().lower()
#         if vs in {'true', 't', '1', 'yes', 'y'}:
#             return True
#         if vs in {'false', 'f', '0', 'no', 'n'}:
#             return False
#         # fallback: treat numeric-like strings
#         if re.fullmatch(r"\d+", vs):
#             return int(vs) != 0
#         return False

#     return s.map(parse_val)


# def collect_accuracies_from_path(path: Path, model_name: str) -> list[dict]:
#     rows = []
#     if not path.exists():
#         return rows

#     for f in sorted(path.iterdir()):
#         if f.is_dir():
#             # maybe nested structure: iterate csv/xlsx inside
#             rows += collect_accuracies_from_path(f, model_name)
#             continue

#         if f.suffix.lower() in {'.xlsx', '.xls'}:
#             try:
#                 sheets = pd.read_excel(f, sheet_name=None)
#             except Exception:
#                 continue
#             for sheet_name, df in sheets.items():
#                 col = find_is_correct_col(df)
#                 if col is None:
#                     continue
#                 lang = sheet_name.lower()
#                 total = len(df)
#                 if total == 0:
#                     acc = 0.0
#                 else:
#                     acc = to_bool_series(df[col]).sum() / total
#                 rows.append({
#                     'model': model_name,
#                     'language': lang,
#                     'accuracy': acc,
#                     'n': total,
#                     'source_file': str(f.name),
#                     'sheet': sheet_name,
#                 })

#         elif f.suffix.lower() == '.csv':
#             try:
#                 df = pd.read_csv(f)
#             except Exception:
#                 continue
#             col = find_is_correct_col(df)
#             if col is None:
#                 continue
#             lang = infer_language_from_name(f.stem)
#             total = len(df)
#             acc = to_bool_series(df[col]).sum() / total if total else 0.0
#             rows.append({
#                 'model': model_name,
#                 'language': lang,
#                 'accuracy': acc,
#                 'n': total,
#                 'source_file': str(f.name),
#                 'sheet': None,
#             })

#     return rows


# def main(cot_dir: Path, direct_dir: Path, out_dir: Path):
#     out_dir.mkdir(parents=True, exist_ok=True)

#     cot_rows = []
#     direct_rows = []

#     # Only process final Excel files named final_data_<model>.xlsx and final_direct_<model>.xlsx
#     cot_final_dir = cot_dir / 'final'
#     direct_final_dir = direct_dir / 'final'

#     LANG_CODES = {'en', 'bn', 'te', 'sw', 'zh'}

#     def extract_from_final_excel(path: Path, model_prefix: str) -> list[dict]:
#         rows = []
#         if not path.exists():
#             return rows
#         for f in sorted(path.iterdir()):
#             if not f.is_file():
#                 continue
#             name = f.name
#             if not name.lower().startswith(model_prefix):
#                 continue
#             model = re.sub(rf'^{re.escape(model_prefix)}', '', Path(name).stem, flags=re.I)
#             try:
#                 sheets = pd.read_excel(f, sheet_name=None)
#             except Exception:
#                 continue
#             for sheet_name, df in sheets.items():
#                 # First, detect any majority-like columns that include language in the name
#                 # e.g., 'cot_majority_en', 'direct_majority_en', 'majority_en', 'majority-correct_en'
#                 maj_cols = []
#                 for col in df.columns:
#                     m = re.search(r'(?:cot_|direct_)?majority(?:[_\-]?correct)?[_\-]?([a-z]{2,3})$', col, re.I)
#                     if m:
#                         lang = m.group(1).lower()
#                         maj_cols.append((col, lang))

#                 if maj_cols:
#                     total = len(df)
#                     for col, lang in maj_cols:
#                         true_count = int(to_bool_series(df[col]).sum()) if total else 0
#                         acc = true_count / total if total else 0.0
#                         rows.append({
#                             'model': model,
#                             'language': lang,
#                             'accuracy': acc,
#                             'true_count': true_count,
#                             'n': total,
#                             'source_file': name,
#                             'sheet': sheet_name,
#                         })
#                     continue

#                 # determine language: prefer sheet name if it matches known codes
#                 lang = sheet_name.lower()
#                 if lang not in LANG_CODES:
#                     # check for column 'language' or 'lang'
#                     for c in ['language', 'lang']:
#                         if c in df.columns:
#                             vals = df[c].dropna().unique()
#                             if len(vals) > 0:
#                                 v = str(vals[0]).lower()
#                                 if v in LANG_CODES:
#                                     lang = v
#                                     break
#                     else:
#                         # try to infer from sheet name tokens like 'en', 'zh'
#                         m = re.search(r'\b(en|bn|te|sw|zh)\b', sheet_name, re.I)
#                         if m:
#                             lang = m.group(1).lower()
#                 col = find_is_correct_col(df)
#                 if col is None:
#                     continue
#                 total = len(df)
#                 true_count = int(to_bool_series(df[col]).sum()) if total else 0
#                 acc = true_count / total if total else 0.0
#                 rows.append({
#                     'model': model,
#                     'language': lang,
#                     'accuracy': acc,
#                     'true_count': true_count,
#                     'n': total,
#                     'source_file': name,
#                     'sheet': sheet_name,
#                 })
#         return rows

#     cot_rows = extract_from_final_excel(cot_final_dir, 'final_data_')
#     direct_rows = extract_from_final_excel(direct_final_dir, 'final_direct_')

#     cot_df = pd.DataFrame(cot_rows)
#     direct_df = pd.DataFrame(direct_rows)

#     # enforce requested language order: english, chinese, bengali, telugu, swahili
#     LANG_ORDER = ['en', 'zh', 'bn', 'te', 'sw']
#     def order_df_by_lang(df: pd.DataFrame) -> pd.DataFrame:
#         if df.empty:
#             return df
#         # normalize language codes to lower-case
#         df['language'] = df['language'].astype(str).str.lower()
#         # sort by a temporary categorical to preserve string dtype
#         cat = pd.Categorical(df['language'], categories=LANG_ORDER, ordered=True)
#         df['_lang_cat'] = cat
#         df = df.sort_values(['model', '_lang_cat'])
#         df.drop(columns=['_lang_cat'], inplace=True)
#         # ensure language stays string
#         df['language'] = df['language'].astype(str)
#         return df
#     cot_df = order_df_by_lang(cot_df)
#     direct_df = order_df_by_lang(direct_df)

#     out_xlsx = out_dir / 'final_accuracies_mgsm.xlsx'

#     # ensure language column ordering and fill missing languages per model
#     with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
#         if not cot_df.empty:
#             cot_df.to_excel(writer, sheet_name='cot_accuracy', index=False)
#         else:
#             pd.DataFrame(columns=['model','language','accuracy','true_count','n','source_file','sheet']).to_excel(writer, sheet_name='cot_accuracy', index=False)

#         if not direct_df.empty:
#             direct_df.to_excel(writer, sheet_name='direct_accuracy', index=False)
#         else:
#             pd.DataFrame(columns=['model','language','accuracy','true_count','n','source_file','sheet']).to_excel(writer, sheet_name='direct_accuracy', index=False)

#         # diff
#         if not cot_df.empty and not direct_df.empty:
#             # normalize language values: extract known lang codes if embedded
#             def extract_lang(val: str) -> str:
#                 if pd.isna(val):
#                     return ''
#                 s = str(val).lower()
#                 m = re.search(r'\b(en|zh|bn|te|sw)\b', s)
#                 if m:
#                     return m.group(1)
#                 # try suffix patterns like _en, -en
#                 m = re.search(r'[_\-]([a-z]{2,3})$', s)
#                 if m:
#                     return m.group(1)
#                 return s

#             cot_df['language'] = cot_df['language'].map(extract_lang)
#             direct_df['language'] = direct_df['language'].map(extract_lang)

#             left = cot_df.rename(columns={'accuracy': 'cot_accuracy'})[['model', 'language', 'cot_accuracy']]
#             right = direct_df.rename(columns={'accuracy': 'direct_accuracy'})[['model', 'language', 'direct_accuracy']]

#             merged = pd.merge(left, right, on=['model', 'language'], how='inner')
#             merged['accuracy_diff'] = merged['cot_accuracy'] - merged['direct_accuracy']

#             # keep only requested three columns
#             diff_df = merged[['model', 'language', 'accuracy_diff']]

#             # order languages and rows
#             diff_df['language'] = diff_df['language'].astype(str).str.lower()
#             diff_df['language'] = pd.Categorical(diff_df['language'], categories=LANG_ORDER, ordered=True)
#             diff_df.sort_values(['model', 'language'], inplace=True)
#             # restore language to string
#             diff_df['language'] = diff_df['language'].astype(str)

#             diff_df.to_excel(writer, sheet_name='cot_vs_direct_diff', index=False)
#         else:
#             pd.DataFrame(columns=['model','language','cot_accuracy','direct_accuracy','accuracy_diff']).to_excel(writer, sheet_name='cot_vs_direct_diff', index=False)

#     print('Wrote:', out_xlsx)


# if __name__ == '__main__':
#     p = argparse.ArgumentParser(
#         description='Compute CoT and Direct accuracies per model and language and export diff CSVs for MGSM.'
#     )
#     dataset_name = 'mgsm'
#     # most result CSVs live under the model folders (e.g. results/cot_inference/mgsm/gemma3)
#     p.add_argument('--cot-dir', type=Path, default=Path(f'results/cot_inference/{dataset_name}'))
#     p.add_argument('--direct-dir', type=Path, default=Path(f'results/direct_inference/{dataset_name}'))
#     p.add_argument('--out-dir', type=Path, default=Path(f'results/hypothesis_1/{dataset_name}'))
#     args = p.parse_args()
#     main(args.cot_dir, args.direct_dir, args.out_dir)

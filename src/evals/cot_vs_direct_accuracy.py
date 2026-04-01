import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_language_from_name(name: str) -> str:
    m = re.search(r'[_\-]([a-z]{2,3})$', name, re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r'[_\-]([a-z]{2,3})\.', name, re.I)
    if m:
        return m.group(1).lower()
    return Path(name).stem


def find_is_correct_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.lower() in {'majority_correct', 'majority-correct', 'majority'}:
            return col
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
        if re.fullmatch(r"\d+", vs):
            return int(vs) != 0
        return False

    return s.map(parse_val)


def extract_lang(val) -> str:
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


LANG_CODES = {'en', 'bn', 'te', 'sw', 'zh'}
LANG_ORDER = ['en', 'zh', 'bn', 'te', 'sw']


# ---------------------------------------------------------------------------
# Core extraction — returns both aggregate stats AND raw bool series for pairing
# ---------------------------------------------------------------------------

def extract_from_final_excel(path: Path, model_prefix: str) -> tuple[list[dict], dict]:
    """
    Returns:
        rows      : list of per-(model, language) aggregate dicts
        raw_index : dict keyed by (model, language) -> pd.Series of bool (reset index)
                    Used later to compute row-level pairing metrics.
    """
    rows = []
    raw_index: dict[tuple[str, str], pd.Series] = {}

    if not path.exists():
        return rows, raw_index

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
            # ---- Case A: majority columns that embed the language in their name ----
            maj_cols = []
            for col in df.columns:
                m = re.search(
                    r'(?:cot_|direct_)?majority(?:[_\-]?correct)?[_\-]?([a-z]{2,3})$',
                    col, re.I
                )
                if m:
                    lang = m.group(1).lower()
                    maj_cols.append((col, lang))

            if maj_cols:
                total = len(df)
                for col, lang in maj_cols:
                    bool_series = to_bool_series(df[col]).reset_index(drop=True)
                    true_count = int(bool_series.sum()) if total else 0
                    acc = true_count / total if total else 0.0
                    rows.append({
                        'model': model, 'language': lang,
                        'accuracy': acc, 'true_count': true_count,
                        'n': total, 'source_file': name, 'sheet': sheet_name,
                    })
                    raw_index[(model, lang)] = bool_series
                continue

            # ---- Case B: standard sheet with a single is_correct-like column ----
            lang = sheet_name.lower()
            if lang not in LANG_CODES:
                for c in ['language', 'lang']:
                    if c in df.columns:
                        vals = df[c].dropna().unique()
                        if len(vals) > 0:
                            v = str(vals[0]).lower()
                            if v in LANG_CODES:
                                lang = v
                                break
                else:
                    m = re.search(r'\b(en|bn|te|sw|zh)\b', sheet_name, re.I)
                    if m:
                        lang = m.group(1).lower()

            col = find_is_correct_col(df)
            if col is None:
                continue

            total = len(df)
            bool_series = to_bool_series(df[col]).reset_index(drop=True)
            true_count = int(bool_series.sum()) if total else 0
            acc = true_count / total if total else 0.0

            rows.append({
                'model': model, 'language': lang,
                'accuracy': acc, 'true_count': true_count,
                'n': total, 'source_file': name, 'sheet': sheet_name,
            })
            raw_index[(model, lang)] = bool_series

    return rows, raw_index


# ---------------------------------------------------------------------------
# Row-level pairing metrics
# ---------------------------------------------------------------------------

def compute_pairing_metrics(
    cot_raw: dict[tuple[str, str], pd.Series],
    direct_raw: dict[tuple[str, str], pd.Series],
) -> list[dict]:
    """
    For every (model, language) pair present in both raw dicts, compute:
      - answer_change_rate     : fraction of questions where CoT answer ≠ Direct answer
      - flip_to_correct_rate   : Direct wrong → CoT right  (reasoning helped)
      - flip_to_wrong_rate     : Direct right → CoT wrong  (reasoning hurt)
      - net_flip_rate          : flip_to_correct − flip_to_wrong
    All rates are expressed as fractions (0–1).
    """
    rows = []
    common_keys = set(cot_raw.keys()) & set(direct_raw.keys())

    for (model, lang) in sorted(common_keys):
        cot_series    = cot_raw[(model, lang)]
        direct_series = direct_raw[(model, lang)]

        # Align on shared indices (handles length mismatches gracefully)
        combined = pd.DataFrame({'cot': cot_series, 'direct': direct_series}).dropna()
        n = len(combined)
        if n == 0:
            continue

        cot_correct    = combined['cot'].astype(bool)
        direct_correct = combined['direct'].astype(bool)

        changed           = cot_correct != direct_correct
        flip_to_correct   = (~direct_correct) & cot_correct   # wrong → right
        flip_to_wrong     = direct_correct & (~cot_correct)    # right → wrong

        rows.append({
            'model': model,
            'language': lang,
            'n_paired': n,
            'answer_change_rate':   changed.sum()         / n,
            'flip_to_correct_rate': flip_to_correct.sum() / n,
            'flip_to_wrong_rate':   flip_to_wrong.sum()   / n,
            'net_flip_rate':        (flip_to_correct.sum() - flip_to_wrong.sum()) / n,
        })

    return rows


# ---------------------------------------------------------------------------
# Ordering helper
# ---------------------------------------------------------------------------

def order_df_by_lang(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df['language'] = df['language'].astype(str).str.lower()
    df['_lang_cat'] = pd.Categorical(df['language'], categories=LANG_ORDER, ordered=True)
    df = df.sort_values(['model', '_lang_cat']).drop(columns=['_lang_cat'])
    df['language'] = df['language'].astype(str)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cot_dir: Path, direct_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    cot_final_dir    = cot_dir    / 'final'
    direct_final_dir = direct_dir / 'final'

    cot_rows,    cot_raw    = extract_from_final_excel(cot_final_dir,    'final_data_')
    direct_rows, direct_raw = extract_from_final_excel(direct_final_dir, 'final_direct_')

    cot_df    = pd.DataFrame(cot_rows)
    direct_df = pd.DataFrame(direct_rows)

    if not cot_df.empty:
        cot_df['language'] = cot_df['language'].map(extract_lang)
    if not direct_df.empty:
        direct_df['language'] = direct_df['language'].map(extract_lang)

    cot_df    = order_df_by_lang(cot_df)
    direct_df = order_df_by_lang(direct_df)

    # ---- Normalise language keys in raw dicts too ----
    cot_raw    = {(m, extract_lang(l)): v for (m, l), v in cot_raw.items()}
    direct_raw = {(m, extract_lang(l)): v for (m, l), v in direct_raw.items()}

    # ---- Build diff sheet (cot_vs_direct_diff) ----
    diff_df = pd.DataFrame()
    if not cot_df.empty and not direct_df.empty:
        left  = cot_df.rename(columns={'accuracy': 'cot_accuracy'})[['model', 'language', 'cot_accuracy']]
        right = direct_df.rename(columns={'accuracy': 'direct_accuracy'})[['model', 'language', 'direct_accuracy']]

        merged = pd.merge(left, right, on=['model', 'language'], how='inner')

        # Raw difference (same as before)
        merged['accuracy_diff'] = merged['cot_accuracy'] - merged['direct_accuracy']

        # Relative improvement: (CoT − Direct) / (1 − Direct)
        # If Direct = 1.0 (perfect ceiling), headroom = 0 → NaN
        headroom = 1.0 - merged['direct_accuracy']
        merged['relative_improvement'] = np.where(
            headroom == 0,
            np.nan,
            merged['accuracy_diff'] / headroom,
        )

        diff_df = merged[['model', 'language', 'accuracy_diff', 'relative_improvement']].copy()
        diff_df = order_df_by_lang(diff_df)

    # ---- Build flip_metrics sheet (row-level pairing) ----
    pairing_rows = compute_pairing_metrics(cot_raw, direct_raw)
    if pairing_rows:
        flip_df = pd.DataFrame(pairing_rows)
        flip_df['language'] = flip_df['language'].map(extract_lang)
        flip_df = order_df_by_lang(flip_df)
        flip_df = flip_df[['model', 'language', 'n_paired',
                            'answer_change_rate', 'flip_to_correct_rate',
                            'flip_to_wrong_rate', 'net_flip_rate']]
    else:
        flip_df = pd.DataFrame(columns=['model', 'language', 'n_paired',
                                        'answer_change_rate', 'flip_to_correct_rate',
                                        'flip_to_wrong_rate', 'net_flip_rate'])

    # ---- Write output ----
    out_xlsx = out_dir / 'final_accuracies_mgsm.xlsx'

    empty_acc_cols  = ['model', 'language', 'accuracy', 'true_count', 'n', 'source_file', 'sheet']
    empty_diff_cols = ['model', 'language', 'accuracy_diff', 'relative_improvement']
    empty_flip_cols = ['model', 'language', 'n_paired', 'answer_change_rate',
                       'flip_to_correct_rate', 'flip_to_wrong_rate', 'net_flip_rate']

    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        # Sheet 1: CoT accuracy (raw) — unchanged
        (cot_df if not cot_df.empty
         else pd.DataFrame(columns=empty_acc_cols)
        ).to_excel(writer, sheet_name='cot_accuracy', index=False)

        # Sheet 2: Direct accuracy (raw) — unchanged
        (direct_df if not direct_df.empty
         else pd.DataFrame(columns=empty_acc_cols)
        ).to_excel(writer, sheet_name='direct_accuracy', index=False)

        # Sheet 3: cot_vs_direct_diff — same as before + relative_improvement column
        (diff_df if not diff_df.empty
         else pd.DataFrame(columns=empty_diff_cols)
        ).to_excel(writer, sheet_name='cot_vs_direct_diff', index=False)

        # Sheet 4: flip_metrics — new, row-level behavioural metrics
        (flip_df if not flip_df.empty
         else pd.DataFrame(columns=empty_flip_cols)
        ).to_excel(writer, sheet_name='flip_metrics', index=False)

    print('Wrote:', out_xlsx)
    if not diff_df.empty:
        print('\nDiff + Normalised preview:')
        print(diff_df.to_string(index=False))
    if not flip_df.empty:
        print('\nFlip metrics preview:')
        print(flip_df.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Compute CoT vs Direct metrics per model and language for MGSM.'
    )
    dataset_name = 'mmlu'
    p.add_argument('--cot-dir',    type=Path, default=Path(f'../../results/cot_inference/{dataset_name}'))
    p.add_argument('--direct-dir', type=Path, default=Path(f'../../results/direct_inference/{dataset_name}'))
    p.add_argument('--out-dir',    type=Path, default=Path(f'../../results/hypothesis_1/{dataset_name}'))
    args = p.parse_args()
    main(args.cot_dir, args.direct_dir, args.out_dir)
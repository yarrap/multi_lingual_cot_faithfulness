"""
MGSM Aggregation Pipeline
Merges per-model scores, imputes missing values, and assigns difficulty labels.
"""
import os
import yaml
import argparse
import numpy as np
import pandas as pd

def load_config(config_path):
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)

def krippendorff_alpha_ordinal(ratings):
    ratings = np.array(ratings, dtype=float)
    flat = ratings.flatten()
    observed = flat[~np.isnan(flat)]
    values = np.sort(np.unique(observed))
    n_values = len(values)
    
    if n_values < 2:
        return np.nan
        
    val_counts = np.array([(observed == v).sum() for v in values], dtype=float)
    dist2 = np.zeros((n_values, n_values))
    for ki in range(n_values):
        for li in range(n_values):
            if ki != li:
                lo, hi = min(ki, li), max(ki, li)
                segment = val_counts[lo : hi + 1]
                d = segment.sum() - (val_counts[lo] + val_counts[hi]) / 2.0
                dist2[ki, li] = d ** 2

    def idx(v): return int(np.searchsorted(values, v))
    
    n_items = ratings.shape[1]
    coincidence = np.zeros((n_values, n_values))
    for item in range(n_items):
        col = ratings[:, item]
        valid = col[~np.isnan(col)]
        m = len(valid)
        if m >= 2:
            weight = 1.0 / (m - 1)
            for i in range(m):
                for j in range(m):
                    if i != j:
                        coincidence[idx(valid[i]), idx(valid[j])] += weight
                        
    n_k = coincidence.sum(axis=1)
    n_total = coincidence.sum()
    D_o = (coincidence * dist2).sum() / n_total
    
    D_e = 0.0
    for ki in range(n_values):
        for li in range(n_values):
            D_e += n_k[ki] * n_k[li] * dist2[ki, li]
    D_e /= n_total * (n_total - 1)
    
    return 1.0 if D_e == 0 and D_o == 0 else (1.0 - D_o / D_e if D_e != 0 else 0)

def build_tertile_bins(series):
    t33, t67 = series.quantile([1/3, 2/3])
    if t33 == t67:
        med = series.median()
        t33, t67 = (series.min() + med) / 2, (med + series.max()) / 2
    return [-np.inf, t33, t67, np.inf], ["easy", "medium", "hard"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/mgsm_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = config["data"]["output_dir"]
    run_order = config["models"]["run_order"]
    
    completed = []
    for name in run_order:
        out_path = os.path.join(output_dir, f"scores_{name.replace('.', '_')}.csv")
        if os.path.exists(out_path):
            completed.append(pd.read_csv(out_path)[["question", "answer_number", f"score_{name}"]])
            
    if not completed:
        print("No completed model CSVs found. Exiting.")
        return
        
    df = completed[0]
    for df_m in completed[1:]:
        df = df.merge(df_m, on=["question", "answer_number"], how="outer")
        
    llm_cols = [f"score_{name}" for name in run_order if f"score_{name}" in df.columns]
    
    for col in llm_cols:
        mask = df[col].isna()
        if mask.any():
            others = [c for c in llm_cols if c != col]
            df.loc[mask, col] = df.loc[mask, others].mean(axis=1)
            
    df["mean_score"] = df[llm_cols].mean(axis=1)
    df["rater_std"] = df[llm_cols].std(axis=1) if len(llm_cols) > 1 else 0.0
    
    if len(llm_cols) > 1:
        alpha = krippendorff_alpha_ordinal(df[llm_cols].to_numpy().T)
        print(f"Krippendorff's α (ordinal) = {alpha:.4f}")
    
    bins, labels = build_tertile_bins(df["mean_score"])
    df["difficulty"] = pd.cut(df["mean_score"], bins=bins, labels=labels, include_lowest=True)
    
    out_file = config["data"]["final_output"]
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Saved aggregated labels to {out_file}")

if __name__ == "__main__":
    main()

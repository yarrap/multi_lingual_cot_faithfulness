"""
MMLU Aggregation Pipeline
Merges model inference results and assigns consensus difficulty.
"""
import os
import yaml
import argparse
import pandas as pd

def load_config(config_path):
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/mmlu_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    input_dir = config["data"]["output_dir"]
    final_output = config["data"]["final_output"]
    model_names = config["models"]["run_order"]
    
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    
    frames, available = {}, []
    for name in model_names:
        path = os.path.join(input_dir, f"mmlu_inference_{name.replace('.', '_').replace('-', '_')}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path).drop_duplicates(subset="sample_id", keep="last")
            df["correct"] = df["correct"].astype(int)
            df.loc[df["predicted"] == "FAIL", "correct"] = 0
            frames[name] = df
            available.append(name)
            
    if not available:
        print(f"No inference files found in {input_dir}.")
        return
        
    meta_cols = ["sample_id", "subject", "subject_category", "answer", "cultural_sensitivity_label", "is_annotated"]
    df = None
    
    for name in available:
        df_m = frames[name][meta_cols + ["predicted", "correct"]].rename(columns={"predicted": f"pred_{name}", "correct": f"correct_{name}"})
        if df is None:
            df = df_m
        else:
            df = df.merge(df_m[["sample_id", f"pred_{name}", f"correct_{name}"]], on="sample_id", how="outer")
            
    correct_cols = [f"correct_{n}" for n in available]
    for col in correct_cols: df[col] = df[col].fillna(0).astype(int)
    
    df["panel_accuracy"] = df[correct_cols].mean(axis=1)
    df["difficulty"] = df["panel_accuracy"].map(lambda pa: "easy" if pa == 1.0 else ("medium" if pa >= 0.67 else "hard"))
    
    output_cols = meta_cols + [f"pred_{n}" for n in available] + correct_cols + ["panel_accuracy", "difficulty"]
    df[output_cols].to_csv(final_output, index=False)
    print(f"Labels saved to {final_output}")

if __name__ == "__main__":
    main()

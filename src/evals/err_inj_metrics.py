import pandas as pd
import numpy as np

# Path to your reinferred results sheets
file_path = "results/hypothesis_2/error_inj_perturbation/mgsm/final_data_error_inj_tiny-aya-global.xlsx"

languages = ["en", "bn", "sw", "te", "zh"]

summary_rows = []

for lang in languages:
    sheet_name = f"cot_majority_{lang}"
    df = pd.read_excel(file_path, sheet_name=sheet_name).fillna("")

    # Ensure numeric
    # Use the actual gold answer column
    df["answer"] = pd.to_numeric(df["answer"], errors="coerce")
    df["final_answer"] = pd.to_numeric(df["final_answer"], errors="coerce") #Cot answer
    df["re_infer_answer"] = pd.to_numeric(df["re_infer_answer"], errors="coerce")
    df["injected_val"] = pd.to_numeric(df["injected_val"], errors="coerce")
    
    # CoT original accuracy (final_answer vs answer)
    df["cot_correct"] = df["final_answer"] == df["answer"]
    cot_accuracy = df["cot_correct"].mean() * 100

    # Re-inference accuracy
    df["reinfer_correct"] = df["re_infer_answer"] == df["answer"]
    reinfer_accuracy = df["reinfer_correct"].mean() * 100
    # basically after cot if the answer is x and then after perturbation if it is y then we will count those
    # Absolute and relative drop
    abs_drop = abs(cot_accuracy - reinfer_accuracy)
    rel_drop = (abs_drop / cot_accuracy * 100) if cot_accuracy > 0 else np.nan

    # Matching ratio: does re_infer_answer match injected value?
    df["match_injected"] = df["re_infer_answer"] == df["injected_val"]
    matching_ratio = df["match_injected"].mean() * 100
    
    df["answer_flip"] = df["final_answer"] != df["re_infer_answer"]
    answer_flip_rate = df["answer_flip"].mean() * 100
    summary_rows.append({
        "language": lang,
        "cot_accuracy (%)": round(cot_accuracy, 2),
        "reinfer_accuracy (%)": round(reinfer_accuracy, 2),
        "absolute_drop (%)": round(abs_drop, 2),
        "relative_drop (%)": round(rel_drop, 2),
        "matching_ratio (%)": round(matching_ratio, 2),
        "answer_flip_rate(%)": round(answer_flip_rate, 2)
    })

    # Optional: save per-sheet updated df with match flags
    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# Summary table
summary_df = pd.DataFrame(summary_rows)
print("\n Faithfulness Summary per Language:\n")
print(summary_df)

# Save summary
summary_df.to_csv(file_path.replace(".xlsx", "_err_inj_metrics.csv"), index=False)
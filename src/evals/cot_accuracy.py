import pandas as pd

df = pd.read_csv("../../results/direct_inference/mgsm/tiny-aya-water/direct_zh.csv")

total = len(df)
true_count = df["is_correct"].sum()
false_count = total - true_count

print(f"Total rows   : {total}")
print(f"True (correct): {true_count}")
print(f"False (wrong) : {false_count}")
print(f"Accuracy      : {true_count/total:.1%}")
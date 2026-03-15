import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.evals.mmlu_cot_pertubation import create_truncation_variants, extract_reasoning_steps

test_cots = [
    # English with numbered list
    """1. First, we need to calculate the total revenue from selling 100 apples at $2 each. Revenue = 100 * 2 = $200.
2. Next, we find the total cost of the apples, which were bought for $1 each. Cost = 100 * 1 = $100.
3. Finally, we subtract the cost from the revenue to get the profit. Profit = $200 - $100 = $100.
Therefore, the profit is $100.""",

    # Chinese with punctuation separators
    """首先，我们需要计算100个苹果每个2美元的总收入。 收入 = 100 * 2 = 200美元。 
其次，我们找到这些苹果的总成本，购买时每个1美元。 成本 = 100 * 1 = 100美元。 
最后，我们从收入中减去成本得出利润。 利润 = 200 - 100 = 100美元。 
因此，利润是100美元。""",

    # English with short, arbitrary sentence breaks
    """The capital of France is Paris. The capital of Spain is Madrid. The capital of Italy is Rome. The capital of Germany is Berlin. The capital of UK is London.""",

    # A short, 2-step reasoning
    """A triangle has 3 sides. A square has 4 sides."""
]

print("--- TESTING TRUNCATION LOGIC ---\n")

for i, cot in enumerate(test_cots):
    print(f"=== Test CoT {i+1} ===")
    print("ORIGINAL:")
    print(f"{cot}\n")
    
    steps = extract_reasoning_steps(cot)
    print(f"EXTRACTED STEPS ({len(steps)}):")
    for j, step in enumerate(steps):
        print(f"  Step {j+1}: {step}")
    print()
    
    variants = create_truncation_variants(cot)
    print("VARIANT: REMOVE FIRST")
    print(f"{variants['remove_first']}\n")
    print("VARIANT: REMOVE LAST")
    print(f"{variants['remove_last']}\n")
    print("-" * 40 + "\n")

import sys
import os
import io

from src.perturbation.mmlu_truncation import create_truncation_variants, extract_reasoning_steps

test_cots = [
    {
        "lang": "en",
        "name": "English Numbered List",
        "text": """1. First, we need to calculate the total revenue from selling 100 apples at $2 each. Revenue = 100 * 2 = $200.
2. Next, we find the total cost of the apples, which were bought for $1 each. Cost = 100 * 1 = $100.
3. Finally, we subtract the cost from the revenue to get the profit. Profit = $200 - $100 = $100.
Therefore, the profit is $100."""
    },
    {
        "lang": "zh",
        "name": "Chinese Punctuation",
        "text": """首先，我们需要计算100个苹果每个2美元的总收入。收入 = 100 * 2 = 200美元。其次，我们找到这些苹果的总成本，购买时每个1美元。成本 = 100 * 1 = 100美元。最后，我们从收入中减去成本得出利润。利润 = 200 - 100 = 100美元。因此，利润是100美元。"""
    },
    {
        "lang": "bn",
        "name": "Bengali Punctuation",
        "text": """১. ধরি রহিমের কাছে ১০০ টাকা আছে। সে বাজার থেকে ২০ টাকার আম কিনল। এখন তার কাছে ৮০ টাকা আছে। এরপর সে ১০ টাকার কলা কিনল। অবশেষে তার কাছে ৭০ টাকা রইল।"""
    },
    {
        "lang": "te",
        "name": "Telugu Numbered List",
        "text": """1. మొదట, ఒక వస్తువు ధర 100 రూపాయలు అనుకుందాం. 2. దానికి 10 శాతం లాభం కలిపితే అది 110 రూపాయలు అవుతుంది. 3. కాబట్టి అమ్మకపు ధర 110 రూపాయలు."""
    },
    {
        "lang": "sw",
        "name": "Swahili Sentences",
        "text": """Kwanza, tunahesabu jumla. Pili, tunatoa gharama. Tatu, tunapata matokeo ya mwisho. Hii ndiyo njia rahisi zaidi."""
    },
    {
        "lang": "en",
        "name": "Double Newline Blocks",
        "text": """This is the first segment of reasoning. It explains the base concept.

Now we move to the second part. This involves the actual calculation phase.

Finally, we reached the conclusion. The result is verified."""
    }
]

output_path = "results/truncation_test_results.txt"
if not os.path.exists("results"):
    os.makedirs("results")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("--- MULTILINGUAL TRUNCATION LOGIC TEST RESULTS ---\n\n")
    
    for i, case in enumerate(test_cots):
        f.write(f"=== Test {i+1}: {case['name']} ({case['lang'].upper()}) ===\n")
        f.write("ORIGINAL:\n")
        f.write(f"{case['text']}\n\n")
        
        steps = extract_reasoning_steps(case['text'])
        f.write(f"EXTRACTED STEPS ({len(steps)}):\n")
        for j, step in enumerate(steps):
            f.write(f"  Step {j+1}: {step}\n")
        f.write("\n")
        
        variants = create_truncation_variants(case['text'])
        f.write("TRUNCATION VARIANTS:\n")
        f.write("--- REMOVE FIRST ---\n")
        f.write(f"{variants['remove_first']}\n\n")
        f.write("--- REMOVE MIDDLE ---\n")
        f.write(f"{variants['remove_middle']}\n\n")
        f.write("--- REMOVE LAST ---\n")
        f.write(f"{variants['remove_last']}\n\n")
        f.write("=" * 60 + "\n\n")

print(f"Truncation test completed. Results saved to {output_path}")

import os
import csv
from pathlib import Path
import run_activity_rounds

csv_file = 'activity_artifacts/round7_confusion_matrix.csv'
out_file = Path('activity_artifacts/round7_per_class_accuracy.png')

CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

cm = []
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        nums = []
        for x in row:
            try:
                nums.append(int(x.strip()))
            except ValueError:
                pass
        if len(nums) == 10:
            cm.append(nums)

lines = []
lines.append(f"{'Class':<12} {'Accuracy':>10} {'Samples':>10}")
lines.append("-" * 34)

total_correct = 0
total_samples = 0

for i, name in enumerate(CLASS_NAMES):
    correct = cm[i][i]
    total = sum(cm[i])
    acc = correct / total if total > 0 else 0
    lines.append(f"{name:<12} {acc*100:>9.1f}% {int(total):>10}")
    total_correct += correct
    total_samples += total

lines.append("-" * 34)
overall_acc = total_correct / total_samples if total_samples > 0 else 0
lines.append(f"{'OVERALL':<12} {overall_acc*100:>9.1f}% {int(total_samples):>10}")

text = "\n".join(lines)

run_activity_rounds.save_text_proof_image(
    text=text,
    output_path=out_file,
    title="Round 7 - Per-Class Accuracy",
    font_size=18
)
print("Saved", out_file)

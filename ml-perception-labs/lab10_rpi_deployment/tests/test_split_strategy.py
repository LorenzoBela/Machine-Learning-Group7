import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_lab10_gpu import build_split_indices


class SplitStrategyTests(unittest.TestCase):
    def test_blocked_split_holds_out_later_images_per_class(self):
        samples = []
        targets = []
        for class_idx, class_name in enumerate(["A", "B"]):
            for image_idx in range(1, 11):
                samples.append((f"{class_name}_{image_idx:03d}.jpg", class_idx))
                targets.append(class_idx)

        train, val, test = build_split_indices(samples, targets, strategy="blocked_by_filename", seed=42)

        self.assertEqual(len(train), 12)
        self.assertEqual(len(val), 4)
        self.assertEqual(len(test), 4)
        self.assertEqual([samples[i][0] for i in test], ["A_009.jpg", "A_010.jpg", "B_009.jpg", "B_010.jpg"])
        self.assertEqual([samples[i][0] for i in val], ["A_007.jpg", "A_008.jpg", "B_007.jpg", "B_008.jpg"])


if __name__ == "__main__":
    unittest.main()

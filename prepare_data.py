import os
from datasets import load_dataset
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────
REAL_TRAIN = "data/cifake/train/REAL"
FAKE_TRAIN = "data/cifake/train/FAKE"
REAL_TEST  = "data/cifake/test/REAL"
FAKE_TEST  = "data/cifake/test/FAKE"

# Create folders
for folder in [REAL_TRAIN, FAKE_TRAIN, REAL_TEST, FAKE_TEST]:
    os.makedirs(folder, exist_ok=True)

print("Downloading dataset... (this will take 2-3 minutes)")

# ── Load Dataset ──────────────────────────────────────────────
# CIFAKE on HuggingFace — real vs AI generated images
# Total size: ~90MB, 60k images
dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="train")

print(f"Total images: {len(dataset)}")

# ── Split and Save ────────────────────────────────────────────
train_count = {"REAL": 0, "FAKE": 0}
test_count  = {"REAL": 0, "FAKE": 0}

TRAIN_LIMIT = 5000   # 5000 real + 5000 fake for training
TEST_LIMIT  = 1000   # 1000 real + 1000 fake for testing

for item in dataset:
    image = item["image"]                    # PIL image
    label = "REAL" if item["label"] == 1 else "FAKE"  # 1=real, 0=fake

    # Save to train or test based on count
    if train_count[label] < TRAIN_LIMIT:
        path = os.path.join(
            REAL_TRAIN if label == "REAL" else FAKE_TRAIN,
            f"{label}_{train_count[label]}.png"
        )
        image.save(path)
        train_count[label] += 1

    elif test_count[label] < TEST_LIMIT:
        path = os.path.join(
            REAL_TEST if label == "REAL" else FAKE_TEST,
            f"{label}_{test_count[label]}.png"
        )
        image.save(path)
        test_count[label] += 1

    # Stop once we have enough
    if all(v >= TRAIN_LIMIT for v in train_count.values()) and \
       all(v >= TEST_LIMIT  for v in test_count.values()):
        break

print(f"\n✅ Done!")
print(f"Train → REAL: {train_count['REAL']} | FAKE: {train_count['FAKE']}")
print(f"Test  → REAL: {test_count['REAL']}  | FAKE: {test_count['FAKE']}")
print(f"Total images saved: {sum(train_count.values()) + sum(test_count.values())}")
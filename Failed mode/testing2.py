import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
MEMORY_FILE = "semantic_brain.json"
ENCODER_WEIGHTS_FILE = "encoder_weights.pth"
DIMENSIONS = 15

# "Not too confident" cutoff: an image whose best niche match is below this is
# treated as Noise (the web app files it as Pending for discover.py to review).
#
# Contrastive (InfoNCE) training never pushes the winning similarity to a fixed
# absolute number, so this is calibrated empirically. On held-out vault images
# (80/20 splits, 3 seeds) the model's WRONG answers averaged 19-38 top similarity
# versus 56-61 for right ones. A cutoff of 30 flags ~13% of genuine images as
# uncertain; 40 flags ~22% (catches more mistakes, but sends more good images to
# review). It does NOT reliably catch out-of-domain images that happen to look
# like a niche (a movie poster scored 38), so retune this after retraining.
CONFIDENCE_THRESHOLD = 30.0

# -------------------------------------------------------------
# 1. 15-D CNN IMAGE ENCODER (Must match training architecture)
# -------------------------------------------------------------
class ImageVectorEncoder(nn.Module):
    """Architecture must stay identical to train.py's, or saved weights won't load."""
    def __init__(self, output_dim=DIMENSIONS):
        super(ImageVectorEncoder, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.projection = nn.Linear(512, output_dim)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        vec = self.projection(feats)
        return F.normalize(vec, p=2, dim=1)

img_encoder = ImageVectorEncoder()
if os.path.exists(ENCODER_WEIGHTS_FILE):
    try:
        img_encoder.load_state_dict(torch.load(ENCODER_WEIGHTS_FILE, map_location="cpu"))
    except RuntimeError:
        print(f"[!] '{ENCODER_WEIGHTS_FILE}' doesn't match the current architecture "
              f"(likely from the old from-scratch CNN) — delete it and "
              f"rerun train.py first.")
        raise
else:
    print(f"[!] Warning: '{ENCODER_WEIGHTS_FILE}' not found — using a fresh random "
          f"encoder that will NOT match your trained word_vectors. Run train.py first.")
img_encoder.eval()

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def encode_image(image_path):
    img = Image.open(image_path).convert('RGB')
    tensor = img_transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = img_encoder(tensor)[0].tolist()
    return [round(v, 4) for v in vec]

# -------------------------------------------------------------
# 2. LOAD BRAIN MEMORY & MATH UTILS
# -------------------------------------------------------------
def load_brain_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return (
                data.get("word_vectors", {}),
                data.get("niches", []),
                data.get("category_mapping", {}),
                data.get("niche_centroids", {})
            )
    return {}, [], {}, {}

def get_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return (dot_product / (mag_a * mag_b)) * 100  # percentage

# -------------------------------------------------------------
# 3. NICHE PROTOTYPES — prefer the image-space centroids saved by train.py
#    (mean embedding of each niche's training images). On held-out vault
#    splits they classified ~90% correctly vs ~72% for the word-average
#    fallback below: single-word vectors are only weakly determined, since
#    training only ever matches the mean of a whole caption.
# -------------------------------------------------------------
def build_niche_prototypes(word_vectors, category_mapping, niche_centroids=None):
    if niche_centroids:
        return {n: c for n, c in niche_centroids.items() if n != "Uncategorized"}

    # Fallback (old semantic_brain.json without centroids): average ALL of a
    # niche's caption vocabulary.
    prototypes = {}
    for niche, words in category_mapping.items():
        if niche == "Uncategorized":
            continue  # not a real class to classify into — exclude it as a competitor
        vecs = [word_vectors[w] for w in words if w in word_vectors]
        if not vecs:
            continue
        avg = [sum(v[i] for v in vecs) / len(vecs) for i in range(DIMENSIONS)]
        prototypes[niche] = avg
    return prototypes

# -------------------------------------------------------------
# 4. CLASSIFICATION — always returns the FULL ranking, never hides it
# -------------------------------------------------------------
def classify_test_image(image_path, prototypes):
    if not os.path.exists(image_path):
        return "File Not Found", 0.0, {}

    img_vec = encode_image(image_path)

    category_scores = {
        niche: round(get_similarity(img_vec, proto), 2)
        for niche, proto in prototypes.items()
    }

    if not category_scores:
        return "Noise", 0.0, {}

    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    top_category, top_score = sorted_categories[0]

    if top_category == "Uncategorized" or top_score < CONFIDENCE_THRESHOLD:
        return "Noise", top_score, category_scores

    return top_category, top_score, category_scores

def main():
    word_vectors, known_niches, category_mapping, niche_centroids = load_brain_memory()

    if not word_vectors:
        print("[-] Error: 'semantic_brain.json' is empty or missing. Train the model first!")
        return

    prototypes = build_niche_prototypes(word_vectors, category_mapping, niche_centroids)
    if not prototypes:
        print("[-] Error: couldn't build any niche prototypes from category_mapping.")
        return

    test_images = [
        "test_image_1.jpeg",
        "test_image_2.jpeg",
        "test_image_3.jpeg",
        "test_image_4.jpeg"
    ]

    print("=" * 60)
    print(" CLEAN CATEGORICAL MULTIMODAL CLASSIFIER ONLINE")
    print("=" * 60)

    for img_file in test_images:
        if not os.path.exists(img_file):
            print(f"\n[Skipped] '{img_file}' not found in directory.")
            continue

        prediction, confidence, all_scores = classify_test_image(img_file, prototypes)

        print(f"\nTesting File : {img_file}")
        ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"Full ranking  : {ranked}")
        if prediction == "Noise":
            print(f"Classification : \u274c NOISE (Confidence: {confidence:.2f}% - Below threshold)")
        else:
            print(f"Classification : \u2705 Clean Category -> '{prediction}' (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    main()

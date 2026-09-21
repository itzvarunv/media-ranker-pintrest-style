import json
import random
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

MEMORY_FILE = "semantic_brain.json"
ENCODER_WEIGHTS_FILE = "encoder_weights.pth"
DIMENSIONS = 15
LEARNING_SPEED = 0.02

# -------------------------------------------------------------
# 1. IMAGE VECTOR ENCODER (CNN -> 15-D Space)
# -------------------------------------------------------------
class ImageVectorEncoder(nn.Module):
    """Frozen pretrained ResNet18 + trainable linear projection into our 15-D
    vector space. Architecture must stay identical to train.py's, or the
    saved weights won't load."""
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
        # Normalize to unit sphere for clean dot product / cosine similarity
        return F.normalize(vec, p=2, dim=1)

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------------------------------------
# 2. VECTOR MATHEMATICS & CONVERGENCE
# -------------------------------------------------------------
def get_similarity(vec_a, vec_b):
    """Calculates dot product similarity between two 15-D vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return (dot_product / (mag_a * mag_b))

def converge_text_and_image(image_vec, text_words, word_vectors):
    """
    Pulls the word vectors closer to the image vector 
    and pushes the image vector towards the title/description context.

    NOTE: this is a cheap one-off nudge for adding a single new upload
    on the fly — it only moves word_vectors and does NOT touch the CNN's
    weights. The encoder itself is now trained with a real contrastive
    loss over the whole vault in train.py; that's what actually makes
    image vectors meaningful. Use this function for quick incremental
    additions between full retrains, not as a substitute for train.py.
    """
    for word in text_words:
        if word not in word_vectors:
            word_vectors[word] = [round(random.uniform(-1.0, 1.0), 4) for _ in range(DIMENSIONS)]
        
        word_vec = word_vectors[word]
        # Shift word vector toward image vector
        for i in range(DIMENSIONS):
            gap = image_vec[i] - word_vec[i]
            word_vec[i] = round(word_vec[i] + (LEARNING_SPEED * gap), 4)
            
    return word_vectors

# -------------------------------------------------------------
# 3. TRAINING & MATCHING API
# -------------------------------------------------------------
encoder = ImageVectorEncoder()
if os.path.exists(ENCODER_WEIGHTS_FILE):
    # Reuse the exact same projection every other script uses, so image
    # vectors made here land in the same 15-D space as the trained word_vectors.
    try:
        encoder.load_state_dict(torch.load(ENCODER_WEIGHTS_FILE, map_location="cpu"))
    except RuntimeError:
        print(f"[!] '{ENCODER_WEIGHTS_FILE}' doesn't match the current architecture "
              f"(likely from the old from-scratch CNN) — delete it and "
              f"rerun train.py first.")
        raise
else:
    # First script to ever run mints the canonical encoder for the whole project.
    torch.save(encoder.state_dict(), ENCODER_WEIGHTS_FILE)
encoder.eval()

def encode_image(image_path):
    """Converts image to 15-D float list."""
    img = Image.open(image_path).convert('RGB')
    tensor = img_transform(img).unsqueeze(0)
    with torch.no_grad():
        vec = encoder(tensor)[0].tolist()
    return [round(v, 4) for v in vec]

def train_upload(image_path, title, description, word_vectors, known_words):
    """
    1. Extracts clean words from title & description.
    2. Encodes the image to a 15-D vector.
    3. Aligns (converges) the image vector with the title/description word vectors.
    """
    text = f"{title} {description}"
    cleaned = "".join([c.lower() if c.isalnum() or c.isspace() else " " for c in text])
    tokens = [w for w in cleaned.split() if known_words.get(w) != "Unacceptable"]

    if not tokens:
        return word_vectors

    # Extract image vector
    img_vec = encode_image(image_path)

    # Shift word vectors toward the image vector (Shared Joint Space)
    word_vectors = converge_text_and_image(img_vec, tokens, word_vectors)
    return word_vectors

def predict_image_matches(image_path, word_vectors, top_n=5):
    """Takes a new image, extracts its vector, and fires the best matching words via dot product."""
    img_vec = encode_image(image_path)
    scores = {}

    for word, w_vec in word_vectors.items():
        score = get_similarity(img_vec, w_vec)
        scores[word] = round(score * 100, 2)

    # Sort by highest dot product alignment
    sorted_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_matches[:top_n]
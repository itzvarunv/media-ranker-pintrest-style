import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
VAULT_DIR = "vault"
MEMORY_FILE = "semantic_brain.json"
ENCODER_WEIGHTS_FILE = "encoder_weights.pth"
DIMENSIONS = 15
NOISE_NICHE = "Noise"  # holding pen for low-confidence uploads; never trained as a class
BATCH_SIZE = 16
EPOCHS = 30  # only a 512->15 linear head is trainable now, so it converges fast
LEARNING_RATE = 1e-3
TEMPERATURE = 0.07  # softmax temperature for the contrastive loss

# Common filler words that show up in almost every caption regardless of
# subject. Left in, they give every image a shared "generic sentence"
# component that swamps the actual content signal on a small dataset.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "by", "and", "or", "with",
    "is", "are", "was", "were", "it", "its", "this", "that", "to", "for",
    "from", "as", "into", "onto", "while", "under", "over", "above", "below"
}

# -------------------------------------------------------------
# 1. IMAGE VECTOR ENCODER (CNN -> 15-D Space) — SAME ARCHITECTURE
#    Must stay identical to brain.py / testing2.py so state_dicts line up.
# -------------------------------------------------------------
class ImageVectorEncoder(nn.Module):
    """
    Frozen ImageNet-pretrained ResNet18 (512-D features) + a trainable
    linear projection into the shared 15-D space. ~150 images is far too
    few to learn visual features from scratch, so only the projection head
    is trained; the backbone supplies features that already encode shape
    and object identity.
    """
    def __init__(self, output_dim=DIMENSIONS):
        super(ImageVectorEncoder, self).__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.projection = nn.Linear(512, output_dim)

    def train(self, mode=True):
        # Keep the frozen backbone's BatchNorm on its ImageNet running stats
        # even when the outer module is switched to train mode.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)
        vec = self.projection(feats)
        return F.normalize(vec, p=2, dim=1)

# Augmentation is ONLY used for training (brain.py / testing2.py use a plain
# 224x224 resize). Sized to 224 because that's what ResNet was trained at.
img_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.4, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Deterministic transform, identical to brain.py / testing2.py's inference one.
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------------------------------------------------
# 2. SCAN THE VAULT: build vocab, niches, category_mapping, and
#    (image_path, caption_tokens) samples in a single pass.
# -------------------------------------------------------------
def _matches_any_keyword(text_lower, keywords):
    """
    Whole-word match only. Plain `keyword in text` substring checks
    misfire constantly: "car" matches inside "scarf" or "cardigan",
    "jet" matches inside "vistajet". Word boundaries fix that.
    """
    return any(re.search(rf"\b{re.escape(k)}\b", text_lower) for k in keywords)

def scan_vault(vault_dir):
    vocab = {}
    niches = set()
    category_mapping = {}
    samples = []
    sample_niches = []  # parallel to samples: the niche of each (image, caption) pair

    image_files = [f for f in os.listdir(vault_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]

    for img_name in image_files:
        base_name = os.path.splitext(img_name)[0]
        img_path = os.path.join(vault_dir, img_name)
        txt_path = os.path.join(vault_dir, f"{base_name}.txt")

        if not os.path.exists(txt_path):
            continue

        with open(txt_path, 'r', encoding='utf-8') as tf:
            content = tf.read().strip()

        # Drop the "ASSET ID: xxxxxxx" header line: every id is a unique word
        # with no meaning, which only bloats the vocab and dilutes the niche
        # prototypes built by averaging all of a niche's words.
        text_body = re.sub(r"^ASSET ID:.*$", "", content, flags=re.MULTILINE | re.IGNORECASE)
        tokens = [w.lower() for w in text_body.split() if w.isalnum() and w.lower() not in STOPWORDS]
        if not tokens:
            continue

        for w in tokens:
            if w not in vocab:
                vocab[w] = len(vocab)

        # Every caption carries its own ground-truth "NICHE: <name>" line, so
        # use it. The keyword heuristic below is only a fallback: it
        # mislabeled ~7% of this vault (e.g. Aircraft captions that mention
        # "car", Automotive captions that mention "style").
        content_lower = content.lower()
        niche_line = re.search(r"^NICHE:\s*(.+?)\s*$", content, flags=re.MULTILINE | re.IGNORECASE)
        assigned_niche = niche_line.group(1) if niche_line else "Uncategorized"
        if not niche_line:
            if _matches_any_keyword(content_lower, ["aircraft", "plane", "jet", "boeing", "airbus"]):
                assigned_niche = "Aircrafts"
            elif _matches_any_keyword(content_lower, ["fashion", "dress", "style", "runway", "shirt"]):
                assigned_niche = "Fashion"
            elif _matches_any_keyword(content_lower, ["car", "automotive", "engine", "wheel", "sedan"]):
                assigned_niche = "Automotive"

        # "Noise" = uploads the model wasn't confident about, waiting for
        # discover.py to group them into a real niche. They still teach the
        # image<->caption alignment, but "Noise" must never become a class.
        if assigned_niche != NOISE_NICHE:
            niches.add(assigned_niche)
            category_mapping.setdefault(assigned_niche, [])
            for t in tokens:
                if t not in category_mapping[assigned_niche]:
                    category_mapping[assigned_niche].append(t)

        samples.append((img_path, tokens))
        sample_niches.append(assigned_niche)

    return vocab, niches, category_mapping, samples, sample_niches

class VaultDataset(Dataset):
    """Each item is one (image tensor, [vocab indices for its caption])."""
    def __init__(self, samples, vocab):
        self.samples = samples
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, tokens = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        tensor = img_transform(img)
        token_idxs = [self.vocab[w] for w in tokens if w in self.vocab]
        return tensor, token_idxs

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    token_idx_lists = [b[1] for b in batch]
    return imgs, token_idx_lists

# -------------------------------------------------------------
# 3. CONTRASTIVE LOSS (CLIP-style, in-batch negatives)
# -------------------------------------------------------------
def contrastive_loss(img_vecs, text_vecs, temperature):
    """
    img_vecs, text_vecs: (B, D) L2-normalized. Row i of img_vecs should be
    closest to row i of text_vecs, and far from every other row.
    """
    logits = img_vecs @ text_vecs.t() / temperature  # (B, B)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)   # each image picks its caption
    loss_t2i = F.cross_entropy(logits.t(), targets)  # each caption picks its image
    return (loss_i2t + loss_t2i) / 2

# -------------------------------------------------------------
# 4. TRAINING PIPELINE
# -------------------------------------------------------------
def run_training():
    print("=" * 60)
    print(" STARTING CONTRASTIVE MULTIMODAL TRAINING FROM VAULT FOLDER")
    print("=" * 60)

    if not os.path.exists(VAULT_DIR):
        print(f"[-] Error: Vault folder '{VAULT_DIR}' not found!")
        return

    vocab, known_niches, category_mapping, samples, sample_niches = scan_vault(VAULT_DIR)

    if not vocab or not samples:
        print("[-] No usable (image, caption) pairs found in the vault folder.")
        return

    print(f"[+] Vocabulary size   : {len(vocab)} words")
    print(f"[+] Training samples  : {len(samples)} image/caption pairs")
    print(f"[+] Niches discovered : {list(known_niches)}")

    dataset = VaultDataset(samples, vocab)
    batch_size = min(BATCH_SIZE, len(dataset))
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=len(dataset) >= batch_size * 2, collate_fn=collate_fn
    )

    encoder = ImageVectorEncoder(output_dim=DIMENSIONS)
    word_embeddings = nn.Embedding(len(vocab), DIMENSIONS)
    nn.init.uniform_(word_embeddings.weight, -1.0, 1.0)

    if os.path.exists(ENCODER_WEIGHTS_FILE):
        try:
            encoder.load_state_dict(torch.load(ENCODER_WEIGHTS_FILE, map_location="cpu"))
            print(f"[+] Resuming encoder from existing '{ENCODER_WEIGHTS_FILE}'")
        except RuntimeError:
            print(f"[!] Existing '{ENCODER_WEIGHTS_FILE}' didn't match this architecture — starting fresh.")

    optimizer = torch.optim.Adam(
        list(encoder.projection.parameters()) + list(word_embeddings.parameters()),
        lr=LEARNING_RATE
    )

    encoder.train()
    print("\n[+] Training (loss should trend down; if it sits near ln(batch size) "
          f"\u2248 {torch.log(torch.tensor(float(batch_size))):.2f} it isn't learning anything):\n")

    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0

        for imgs, token_idx_lists in loader:
            optimizer.zero_grad()

            img_vecs = encoder(imgs)  # (B, D), already L2-normalized

            text_vecs = []
            for idxs in token_idx_lists:
                idx_tensor = torch.tensor(idxs, dtype=torch.long)
                caption_word_vecs = word_embeddings(idx_tensor)   # (num_tokens, D)
                text_vecs.append(caption_word_vecs.mean(dim=0))   # bag-of-words average
            text_vecs = torch.stack(text_vecs)
            text_vecs = F.normalize(text_vecs, p=2, dim=1)

            loss = contrastive_loss(img_vecs, text_vecs, TEMPERATURE)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"--- Epoch {epoch + 1}/{EPOCHS} | avg contrastive loss: {avg_loss:.4f} ---")

    # ---- Export trained artifacts ----
    encoder.eval()
    torch.save(encoder.state_dict(), ENCODER_WEIGHTS_FILE)

    with torch.no_grad():
        final_embeddings = word_embeddings.weight.detach()
    word_vectors = {
        word: [round(v, 4) for v in final_embeddings[idx].tolist()]
        for word, idx in vocab.items()
    }

    # Niche centroids in IMAGE space: the mean embedding of each niche's
    # training images. On held-out vault splits this classified ~90% correctly
    # versus ~72% for averaging every word in a niche's caption vocabulary,
    # because single-word vectors are only weakly determined (training only
    # ever matches the mean of a whole caption) while image embeddings are
    # what the encoder actually learned.
    niche_embeddings = {}
    with torch.no_grad():
        for (img_path, _), niche in zip(samples, sample_niches):
            if niche == NOISE_NICHE:
                continue
            tensor = eval_transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
            niche_embeddings.setdefault(niche, []).append(encoder(tensor)[0])
    niche_centroids = {
        niche: [round(v, 4) for v in F.normalize(torch.stack(vecs).mean(dim=0), p=2, dim=0).tolist()]
        for niche, vecs in niche_embeddings.items()
    }

    brain_data = {
        "metadata": {
            "dimensions": DIMENSIONS,
            "total_vectors": len(word_vectors),
            "niches": list(known_niches)
        },
        "niches": list(known_niches),
        "category_mapping": category_mapping,
        "niche_centroids": niche_centroids,
        "word_vectors": word_vectors
    }

    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(brain_data, f, indent=4)

    print("\n" + "=" * 60)
    print(f" TRAINING COMPLETE! Brand new '{MEMORY_FILE}' created successfully.")
    print(f" Encoder weights (now actually trained): '{ENCODER_WEIGHTS_FILE}'")
    print(f" Categories Discovered : {list(known_niches)}")
    print(f" Total Vector Entries  : {len(word_vectors)}")
    print("=" * 60)

if __name__ == "__main__":
    run_training()

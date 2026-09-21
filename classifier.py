"""
Bridge from the web app to the trained CLIP-style model.

The working PyTorch code lives in "Failed mode/" (note the space, so it can't be
imported as a package), and the top-level brain.py / testing2.py are an
unrelated TensorFlow attempt with the same names. So the model module is loaded
by file path under a private name instead of `import testing2`.

To point the app at a different folder later (e.g. after renaming "Failed mode"
to "ml"), change ML_DIR below - nothing else needs to move.
"""
import importlib.util
import os
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, "Failed mode")
WEIGHTS_FILE = os.path.join(BASE_DIR, "encoder_weights.pth")
MEMORY_FILE = os.path.join(BASE_DIR, "semantic_brain.json")

_lock = threading.Lock()
_module = None
_prototypes = None
_memory_mtime = None      # when semantic_brain.json was last read (discover.py rewrites it)


class ClassifierUnavailable(RuntimeError):
    """The model files are missing or failed to load."""


def _read_prototypes(module):
    global _memory_mtime
    module.MEMORY_FILE = MEMORY_FILE
    word_vectors, _niches, category_mapping, niche_centroids = module.load_brain_memory()
    prototypes = module.build_niche_prototypes(word_vectors, category_mapping, niche_centroids)
    if not prototypes:
        raise ClassifierUnavailable("semantic_brain.json has no niche prototypes - retrain with train.py.")
    _memory_mtime = os.path.getmtime(MEMORY_FILE)
    return prototypes


def _refresh_if_changed():
    """Pick up niches that discover.py (a separate process) added, without restarting the app."""
    global _prototypes
    if _module is not None and os.path.getmtime(MEMORY_FILE) != _memory_mtime:
        _prototypes = _read_prototypes(_module)


def _load():
    global _module, _prototypes
    if _module is not None:
        return
    for path in (WEIGHTS_FILE, MEMORY_FILE):
        if not os.path.exists(path):
            raise ClassifierUnavailable(
                f"'{os.path.basename(path)}' not found - run train.py from the Neuroed folder first."
            )
    script = os.path.join(ML_DIR, "testing2.py")
    spec = importlib.util.spec_from_file_location("neuroed_ml_testing2", script)
    module = importlib.util.module_from_spec(spec)
    # testing2.py loads encoder_weights.pth relative to the CWD at import time,
    # so run the import from the project folder (done once, at startup).
    previous_cwd = os.getcwd()
    os.chdir(BASE_DIR)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ClassifierUnavailable(f"Could not load the model: {exc}") from exc
    finally:
        os.chdir(previous_cwd)

    _prototypes = _read_prototypes(module)
    _module = module


def warmup():
    """Load the model now (called at startup so no request pays for it). Returns True/False."""
    with _lock:
        try:
            _load()
            return True
        except ClassifierUnavailable as exc:
            print(f"[!] Classifier unavailable: {exc}")
            return False


def niches():
    with _lock:
        _load()
        _refresh_if_changed()
        return sorted(_prototypes)


def features(image_path):
    """512-D ResNet features (before the 15-D projection) - what discover.py clusters on."""
    from PIL import Image
    import torch
    with _lock:
        _load()
        tensor = _module.img_transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            return _module.img_encoder.backbone(tensor)[0].numpy()


def embedding(image_path):
    """The image's 15-D vector in the shared space (what niche centroids are made of)."""
    with _lock:
        _load()
        return _module.encode_image(image_path)


def classify(image_path):
    """
    Returns {'niche', 'confidence', 'scores', 'rejected'}.
    `rejected` is True when the best match is below testing2's CONFIDENCE_THRESHOLD,
    i.e. the model is "not too confident" - such images become Noise (see feed.py).
    """
    with _lock:
        _load()
        _refresh_if_changed()
        prediction, confidence, scores = _module.classify_test_image(image_path, _prototypes)
    return {
        "niche": prediction,
        "confidence": confidence,
        "scores": scores,
        # The niche with the best score even when the verdict is "Noise" (where `niche` becomes "Noise").
        "best_niche": max(scores, key=scores.get) if scores else None,
        "rejected": prediction in ("Noise", "File Not Found"),
    }

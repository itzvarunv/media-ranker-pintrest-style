"""
Niche discovery for images - the image twin of scanner.py.

scanner.py:  rejected text posts pile up in Noise/, words that co-occur across
             several of them are grouped, you approve the group, and it becomes
             a niche in signals.json.
discover.py: uploads the classifier was "not too confident" about are filed as
             Noise (ranker.genre = 'Noise', status = 'Pending'; see app.py). This script

  1. groups them two ways. By WORDS, like scanner.py: a term that's common in
     the pool but rare in the rest of the vault ("verstappen") seeds a group of
     the posts using it - this is what finds brand / driver / topic niches, which
     look like their parent niche. By LOOKS: the rest are clustered on a blend of
     image features and caption similarity, which finds visually distinct niches.
  2. names each group from caption words that are common INSIDE it but rare in
     the rest of the vault, graded 3/2/1 like scanner.py's signature words
  3. asks you: y / n / or type a different name
  4. on yes: relabels the images (database + the NICHE: line of each caption)
     and adds a centroid to semantic_brain.json, so the app can classify the
     new niche immediately. The next train.py run trains it as a real niche.

Run from the Neuroed folder:
    python3 discover.py              # review and approve proposals
    python3 discover.py --dry-run    # just show what it would propose
"""
import argparse
import json
import os
import re
import tempfile
from collections import Counter, defaultdict

import math

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

import classifier
import vault
from Connector import db_session

NOISE_NICHE = "Noise"
PENDING = "Pending"

MIN_CLUSTER_SIZE = 5       # images a group needs before it's proposed as a niche
IMAGE_WEIGHT = 0.5         # blend of image vs caption distance (captions carry brands/people, images carry looks)
MAX_WORD_SHARE = 0.25      # ignore caption words found in more than this share of the vault (generic prose)
MERGE_DISTANCE = 0.85      # average-linkage distance: groups further apart than this stay separate
MIN_COHESION = 0.15        # mean pairwise similarity inside a group (1 - mean distance)
LOOK_MIN_SIZE = 8          # look-alike-only groups need more images: small visual groups of 5-6 were mostly noise
MIN_LIFT = 10.0            # a term must be this many times more common in the pool than in the rest of the vault
                           # (filler words like 'featuring' reach ~6; a genuinely new term like 'porsche'/'verstappen' is far higher)
MERGE_OVERLAP = 0.6        # two term-groups sharing this much of the smaller one are the same niche (scanner.py's co-occurrence merge)
MIN_IN_CLUSTER = 0.4       # a signature word must appear in at least this share of the group's captions...
MIN_GAP = 0.3              # ...and exceed its share in the rest of the vault by at least this much

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 &'-]{1,39}$")
RESERVED_NAMES = {"noise", "uncategorized"}
WORD_RE = re.compile(r"[a-z][a-z'-]{2,}")


# -------------------------------------------------------------
# 1. THE POOL
# -------------------------------------------------------------
def load_pool():
    """(id, filepath) of every image waiting in Noise."""
    with db_session() as (db, cursor):
        cursor.execute("SELECT id, filepath FROM ranker WHERE genre = %s AND status = %s ORDER BY upload_date",
                       (NOISE_NICHE, PENDING))
        return cursor.fetchall()


def existing_niches():
    with db_session() as (db, cursor):
        cursor.execute("SELECT DISTINCT genre FROM ranker WHERE genre <> %s", (NOISE_NICHE,))
        return {row[0] for row in cursor.fetchall()}


def all_asset_ids():
    with db_session() as (db, cursor):
        cursor.execute("SELECT id FROM ranker")
        return [row[0] for row in cursor.fetchall()]


# -------------------------------------------------------------
# 2. GROUPING BY LOOKS
# -------------------------------------------------------------
def _normalise(features):
    """
    Centre, then L2-normalise. ResNet features are all >= 0, so raw cosine
    similarity is high between ANY two images; subtracting the pool's mean
    removes that shared component and leaves what actually differs.
    """
    X = np.asarray(features, dtype=float)
    X = X - X.mean(axis=0)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def _text_vectors(word_sets, background_sets):
    """TF-IDF vectors of caption words; words that are common across the whole vault carry no weight."""
    n = max(len(background_sets), 1)
    doc_freq = Counter(w for words in background_sets for w in words)
    vocab = sorted({w for words in word_sets for w in words if doc_freq[w] / n <= MAX_WORD_SHARE})
    index = {w: i for i, w in enumerate(vocab)}
    vectors = np.zeros((len(word_sets), max(len(vocab), 1)))
    for row, words in enumerate(word_sets):
        for w in words:
            if w in index:
                vectors[row, index[w]] = math.log((n + 1) / (doc_freq[w] + 1)) + 1
    norms = np.linalg.norm(vectors, axis=1)
    has_text = norms > 0
    vectors[has_text] /= norms[has_text, None]
    return vectors, has_text


def combined_distance(features, word_sets=None, background_sets=None, image_weight=IMAGE_WEIGHT):
    """
    Pairwise distance between pool images: a blend of how they LOOK (cosine on
    centred ResNet features) and what their captions SAY (TF-IDF cosine).
    Looks alone group by scene style (it won't find "Porsche" inside Automotive);
    captions alone need good text. If either image has no caption words, the
    pair falls back to looks only.
    """
    X = _normalise(features)
    image_dist = np.clip(1 - X @ X.T, 0, 2)
    dist = image_dist
    if word_sets is not None:
        vectors, has_text = _text_vectors(word_sets, background_sets if background_sets is not None else word_sets)
        text_dist = np.clip(1 - vectors @ vectors.T, 0, 2)
        both = np.outer(has_text, has_text)
        dist = np.where(both, image_weight * image_dist + (1 - image_weight) * text_dist, image_dist)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    return dist


def cluster_images(features, word_sets=None, background_sets=None, min_size=MIN_CLUSTER_SIZE,
                   max_distance=MERGE_DISTANCE, image_weight=IMAGE_WEIGHT):
    """
    Returns [(row_indices, cohesion), ...] for groups of at least `min_size`
    mutually similar posts, biggest first. Cohesion = 1 - the group's mean
    pairwise distance.
    """
    if len(features) < max(min_size, 2):
        return []
    dist = combined_distance(features, word_sets, background_sets, image_weight)
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"),
                      t=max_distance, criterion="distance")
    groups = defaultdict(list)
    for row, label in enumerate(labels):
        groups[label].append(row)

    found = []
    for rows in groups.values():
        if len(rows) < min_size:
            continue
        sub = dist[np.ix_(rows, rows)]
        cohesion = 1 - float(sub.sum() / (len(rows) * (len(rows) - 1)))
        found.append((rows, cohesion))
    return sorted(found, key=lambda g: len(g[0]), reverse=True)


def term_groups(pool_words, outside_words, min_size=MIN_CLUSTER_SIZE):
    """
    Scanner-style grouping, seeded by WORDS. A term that is common among the
    pool but rare in the rest of the vault (e.g. "verstappen": in 8 new posts,
    in none of the old ones) seeds a group of the posts that use it - the way
    scanner.py builds a niche out of words that co-occur in the same files.
    Groups that mostly contain the same posts are merged, keeping the larger.
    Returns [(row_indices, seed_words), ...], biggest first.
    """
    n_pool, n_out = len(pool_words), max(len(outside_words), 1)
    in_counts = Counter(w for words in pool_words for w in words)
    out_counts = Counter(w for words in outside_words for w in words)

    seeds = []
    for word, count in in_counts.items():
        if count < min_size:
            continue
        lift = (count / n_pool) / ((out_counts[word] + 1) / (n_out + 1))   # +1: "never seen" is not "infinitely rare"
        if lift >= MIN_LIFT:
            seeds.append((count, lift, word))
    seeds.sort(reverse=True)

    groups = []                                   # [set_of_rows, [seed words]]
    for _, _, word in seeds:
        rows = {r for r, words in enumerate(pool_words) if word in words}
        for group in groups:
            if len(rows & group[0]) / min(len(rows), len(group[0])) >= MERGE_OVERLAP:
                group[1].append(word)
                break
        else:
            groups.append([rows, [word]])
    return sorted(([sorted(rows), words] for rows, words in groups), key=lambda g: len(g[0]), reverse=True)


# -------------------------------------------------------------
# 3. NAMING FROM CAPTIONS (scanner.py's graded signature words)
# -------------------------------------------------------------
def caption_words(asset_id):
    caption = vault.read_caption(asset_id)
    words = WORD_RE.findall((caption["title"] + " " + caption["description"]).lower())
    return {w[:-2] if w.endswith("'s") else w for w in words}


def signature_words(member_words, background_words):
    """
    [(word, grade)] best first. A word qualifies when it appears in many of the
    group's captions but few of the rest of the vault's - that is what makes
    "verstappen" a signature word while "captures" or "background" are not.
    Grade 3/2/1 = appears in >=90% / >=60% / fewer of the group's captions.
    """
    if not member_words:
        return []
    inside = Counter(w for words in member_words for w in words)
    outside = Counter(w for words in background_words for w in words)
    n_in, n_out = len(member_words), max(len(background_words), 1)

    scored = {}
    for word, count in inside.items():
        p_in, p_out = count / n_in, outside[word] / n_out
        if p_in >= MIN_IN_CLUSTER and p_in - p_out >= MIN_GAP:
            scored[word] = (p_in - p_out, p_in)
    ranked = sorted(scored, key=lambda w: scored[w], reverse=True)
    return [(w, 3 if scored[w][1] >= 0.9 else 2 if scored[w][1] >= 0.6 else 1) for w in ranked]


def suggest_name(signature, taken, number):
    top = [w for w, _ in signature[:2]]
    base = " & ".join(w.capitalize() for w in top) if top else f"New Niche {number}"
    name, n = base, 2
    while name.lower() in {t.lower() for t in taken}:
        name, n = f"{base} {n}", n + 1
    return name


# -------------------------------------------------------------
# 4. PROMOTING A GROUP TO A NICHE
# -------------------------------------------------------------
def _relabel_caption(asset_id, niche):
    path = vault.caption_path(asset_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return False
    new_text, hits = re.subn(r"^NICHE:.*$", f"NICHE: {niche}", text, count=1, flags=re.MULTILINE)
    if not hits:
        return False
    _atomic_write(path, new_text)
    return True


def _atomic_write(path, text):
    """Write to a temp file in the same folder, then swap it in - never leaves a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _add_centroid(niche, centroid, keywords):
    with open(classifier.MEMORY_FILE, "r", encoding="utf-8") as f:
        brain = json.load(f)
    brain.setdefault("niche_centroids", {})[niche] = [round(float(v), 4) for v in centroid]
    for key_holder in (brain, brain.setdefault("metadata", {})):
        niches = key_holder.setdefault("niches", [])
        if niche not in niches:
            niches.append(niche)
    brain.setdefault("category_mapping", {})[niche] = keywords
    _atomic_write(classifier.MEMORY_FILE, json.dumps(brain, indent=4))


def promote(niche, asset_ids, keywords):
    """
    Turn `asset_ids` (currently Noise) into the live niche `niche`.
    Database + caption files change together (a failure rolls the database back);
    the centroid is written last, atomically.
    """
    embeddings = np.array([classifier.embedding(vault.image_path(i)) for i in asset_ids], dtype=float)
    centroid = embeddings.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-9)

    marks = ", ".join(["%s"] * len(asset_ids))
    with db_session() as (db, cursor):
        cursor.execute(
            f"UPDATE ranker SET genre = %s, status = 'Active' "
            f"WHERE genre = %s AND status = %s AND id IN ({marks})",
            (niche, NOISE_NICHE, PENDING, *asset_ids))
        for asset_id in asset_ids:
            _relabel_caption(asset_id, niche)
    _add_centroid(niche, centroid, keywords)


# -------------------------------------------------------------
# 5. THE INTERACTIVE LOOP
# -------------------------------------------------------------
def build_proposals(ids, features, words, everything, taken):
    """
    The pure core of discovery (no database or files): given the pool's ids and
    image features, every asset's caption words, all asset ids and the niche
    names already in use, return the proposed niches.
    """
    pool_set = set(ids)
    outside = [words[i] for i in everything if i not in pool_set]
    pool_words = [words[i] for i in ids]
    all_words = [words[i] for i in everything]
    dist = combined_distance(features, pool_words, all_words)

    def cohesion_of(rows):
        sub = dist[np.ix_(rows, rows)]
        return 1 - float(sub.sum() / (len(rows) * (len(rows) - 1)))

    candidates, claimed = [], set()                # (rows, cohesion, how it was found)
    for rows, seeds in term_groups(pool_words, outside):
        candidates.append((rows, cohesion_of(rows), "words: " + ", ".join(seeds[:3])))
        claimed.update(rows)
    for rows, cohesion in cluster_images(features, pool_words, all_words):
        fresh = [r for r in rows if r not in claimed]
        if len(fresh) >= LOOK_MIN_SIZE and cohesion >= MIN_COHESION:
            candidates.append((fresh, cohesion, "look-alikes"))
            claimed.update(fresh)

    proposals = []
    for number, (rows, cohesion, source) in enumerate(candidates, start=1):
        members = [ids[r] for r in rows]
        member_set = set(members)
        signature = signature_words([words[i] for i in members],
                                    [words[i] for i in everything if i not in member_set])
        name = suggest_name(signature, taken, number)
        taken = taken | {name}
        proposals.append({"name": name, "ids": members, "cohesion": cohesion,
                          "signature": signature, "source": source})
    return proposals


def find_proposals():
    """Load the Noise pool and describe each group. Returns a list of proposal dicts."""
    pool = load_pool()
    print(f"Noise pool: {len(pool)} image(s) waiting.")
    if len(pool) < MIN_CLUSTER_SIZE:
        print(f"Need at least {MIN_CLUSTER_SIZE} to look for a niche.")
        return []

    ids = [asset_id for asset_id, _ in pool]
    print("Embedding images...")
    features = [classifier.features(vault.image_path(asset_id)) for asset_id in ids]
    everything = all_asset_ids()
    words = {asset_id: caption_words(asset_id) for asset_id in set(everything) | set(ids)}
    proposals = build_proposals(ids, features, words, everything, existing_niches())
    for proposal in proposals:
        # Which existing niche do these images already resemble most? A group that is
        # really a sub-niche (e.g. "Verstappen" inside Automotive) will say so here.
        votes = Counter(classifier.classify(vault.image_path(i))["best_niche"] for i in proposal["ids"])
        proposal["parent"] = votes.most_common(1)[0]
    return proposals


def _describe(proposal):
    words = ", ".join(f"{w}({g})" for w, g in proposal["signature"][:8]) or "(no distinctive caption words)"
    print(f"\n✨ POTENTIAL NICHE: {proposal['name']}  "
          f"({len(proposal['ids'])} images, cohesion {proposal['cohesion']:.2f}, found by {proposal['source']})")
    print(f"   signature words: {words}")
    for asset_id in proposal["ids"][:4]:
        print(f"   - {asset_id}: {vault.read_caption(asset_id)['title'][:70]}")
    parent = proposal.get("parent")
    if parent and parent[1] >= len(proposal["ids"]) / 2:
        print(f"   ⚠ {parent[1]} of {len(proposal['ids'])} of these already classify as '{parent[0]}', so this looks like a "
              f"SUB-niche of it. Approving makes a sibling niche, and similar '{parent[0]}' images may start "
              f"landing in it (more so before you retrain).")


def _valid_name(name, taken):
    if not NAME_RE.match(name) or name.lower() in RESERVED_NAMES:
        return "Use 2-40 letters/numbers (spaces, & ' - allowed); 'Noise' is reserved."
    if name.lower() in {t.lower() for t in taken}:
        return f"'{name}' already exists."
    return None


def run(dry_run=False, ask=input):
    """Propose and (after approval) promote niches. Returns the names that were created."""
    if not classifier.warmup():
        print("Classifier unavailable - run train.py first.")
        return []
    proposals = find_proposals()
    if not proposals:
        print("No new niche found yet.")
        return []

    created, taken = [], existing_niches()
    for proposal in proposals:
        _describe(proposal)
        if dry_run:
            continue
        answer = ask("   Add it as a niche? [y] yes  [n] no  or type a different name: ").strip()
        if answer.lower() in ("", "n", "no"):
            print("   ignored.")
            continue
        name = proposal["name"] if answer.lower() in ("y", "yes") else answer
        problem = _valid_name(name, taken)
        if problem:
            print(f"   not added: {problem}")
            continue
        promote(name, proposal["ids"], [w for w, _ in proposal["signature"]])
        taken.add(name)
        created.append(name)
        print(f"   ✅ '{name}' is live: {len(proposal['ids'])} images moved out of Noise.")

    if created:
        print("\nThe running app picks the new niche(s) up automatically. "
              "Run train.py when convenient to train them properly.")
    return created


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find new niches among low-confidence uploads.")
    parser.add_argument("--dry-run", action="store_true", help="show proposals without changing anything")
    run(dry_run=parser.parse_args().dry_run)

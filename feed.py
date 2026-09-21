"""
The personalised feed and the like / dislike / save interactions.

Feed slots are filled from three sources, following the design in Connector.py:
  * interest      - a genre drawn in proportion to the user's user_interests points
  * discovery     - an item from outside the user's strongest genre(s)
  * collaborative - something a "peer" (collaborative_map) liked
An item counts as *seen* once a user_history row exists for it, so serving an
item creates that row - which is what stops the same post coming back.
"""
import os
import random

import Settings
import vault
from Connector import (db_session, fetch_collaborative_data, fetch_discovery_data,
                       generate_id)

INTEREST_POINTS = {"like": 10, "dislike": -10, "save": 5}   # unsave (-5) lives in Settings
NOISE = "Noise"          # genre of uploads the model wasn't confident about
PENDING = "Pending"      # ranker.status of those uploads: hidden from feeds until discover.py places them
PEER_LIMIT = 5
DISCOVERY_SHARE = 0.15
COLLABORATIVE_SHARE = 0.15


# -------------------------------------------------------------
# CARDS
# -------------------------------------------------------------
def _states(cursor, user_id, ids):
    if not ids:
        return {}
    marks = ", ".join(["%s"] * len(ids))
    cursor.execute(
        f"SELECT blog_id, is_liked, is_disliked, is_saved FROM user_history "
        f"WHERE user_id = %s AND blog_id IN ({marks})", (user_id, *ids))
    return {r[0]: (bool(r[1]), bool(r[2]), bool(r[3])) for r in cursor.fetchall()}


def _cards(user_id, rows):
    """rows: iterable of (id, genre, filepath) -> JSON-ready dicts with the user's own like/save state."""
    rows = list(rows)
    with db_session() as (db, cursor):
        states = _states(cursor, user_id, [r[0] for r in rows])
    cards = []
    for blog_id, genre, filepath in rows:
        liked, disliked, saved = states.get(blog_id, (False, False, False))
        caption = vault.read_caption(blog_id)
        cards.append({
            "id": blog_id,
            "niche": genre,
            "title": caption["title"],
            "description": caption["description"],
            "image_url": f"/vault/{os.path.basename(filepath)}",
            "liked": liked, "disliked": disliked, "saved": saved,
            "pending": genre == NOISE,
        })
    return cards


# -------------------------------------------------------------
# HISTORY / INTERESTS
# -------------------------------------------------------------
def _get_or_create_history(cursor, user_id, blog_id):
    """Returns (history_id, liked, disliked, saved, created)."""
    cursor.execute(
        "SELECT history_id, is_liked, is_disliked, is_saved FROM user_history "
        "WHERE user_id = %s AND blog_id = %s LIMIT 1", (user_id, blog_id))
    row = cursor.fetchone()
    if row:
        return row[0], bool(row[1]), bool(row[2]), bool(row[3]), False
    history_id = generate_id(10)
    cursor.execute("INSERT INTO user_history (history_id, user_id, blog_id) VALUES (%s, %s, %s)",
                   (history_id, user_id, blog_id))
    return history_id, False, False, False, True


def _bump_interest(cursor, user_id, genre, delta):
    cursor.execute("SELECT interest_id, points FROM user_interests WHERE user_id = %s AND genre = %s",
                   (user_id, genre))
    row = cursor.fetchone()
    if row:
        cursor.execute("UPDATE user_interests SET points = %s WHERE interest_id = %s",
                       (max(0.0, row[1] + delta), row[0]))
    else:   # new rows start from the column default of 50
        cursor.execute("INSERT INTO user_interests (interest_id, user_id, genre, points) VALUES (%s, %s, %s, %s)",
                       (generate_id(10), user_id, genre, max(0.0, 50.0 + delta)))


def _mark_viewed(user_id, blog_id):
    with db_session() as (db, cursor):
        *_, created = _get_or_create_history(cursor, user_id, blog_id)
        if created:
            cursor.execute("UPDATE ranker SET views = views + 1 WHERE id = %s", (blog_id,))


# -------------------------------------------------------------
# PICKING
# -------------------------------------------------------------
def _genre_weights(user_id):
    with db_session() as (db, cursor):
        cursor.execute("SELECT DISTINCT genre FROM ranker WHERE status = 'Active'")
        genres = [r[0] for r in cursor.fetchall()]
        cursor.execute("SELECT genre, points FROM user_interests WHERE user_id = %s", (user_id,))
        points = dict(cursor.fetchall())
    # A genre the user has never touched starts at the same 50 a new interest row gets.
    return {g: max(0.0, float(points.get(g, 50.0))) + 1.0 for g in genres}


def _pick_interest(user_id, genre):
    with db_session() as (db, cursor):
        cursor.execute("""
            SELECT r.id, r.genre, r.filepath FROM ranker r
            LEFT JOIN user_history h ON r.id = h.blog_id AND h.user_id = %s
            WHERE r.genre = %s AND r.status = 'Active' AND h.blog_id IS NULL
            ORDER BY (r.likes - r.dislikes) DESC, RAND() LIMIT 1
        """, (user_id, genre))
        return cursor.fetchone()


def _peers(user_id):
    with db_session() as (db, cursor):
        cursor.execute("SELECT peer_id FROM collaborative_map WHERE user_id = %s "
                       "ORDER BY similarity_score DESC LIMIT %s", (user_id, PEER_LIMIT))
        return [r[0] for r in cursor.fetchall()]


def _recycle(user_id, exclude_ids, count):
    """Once everything unseen is used up, replay seen posts the user didn't dislike."""
    if count <= 0:
        return []
    exclude = list(exclude_ids) or ["_none_"]
    marks = ", ".join(["%s"] * len(exclude))
    with db_session() as (db, cursor):
        cursor.execute(f"""
            SELECT r.id, r.genre, r.filepath FROM ranker r
            JOIN user_history h ON r.id = h.blog_id AND h.user_id = %s
            WHERE r.status = 'Active' AND h.is_disliked = 0 AND r.id NOT IN ({marks})
            ORDER BY RAND() LIMIT %s
        """, (user_id, *exclude, count))
        return cursor.fetchall()


def get_feed(user_id, limit=12):
    """Up to `limit` cards for the home tab (each newly-served item is recorded as viewed)."""
    weights = _genre_weights(user_id)
    if not weights:
        return []
    genres = list(weights)
    peers = _peers(user_id)
    exclude_top = min(3, max(1, len(genres) - 2))    # 3 niches -> discover outside the single top one

    picked, picked_ids = [], set()

    def take(row):
        if row and row[0] not in picked_ids:
            picked.append(row); picked_ids.add(row[0])
            _mark_viewed(user_id, row[0])            # so the next pick can't return it again
            return True
        return False

    for _ in range(limit):
        roll = random.random()
        row = None
        if roll < DISCOVERY_SHARE:
            row = fetch_discovery_data(user_id, exclude_top=exclude_top)
        elif roll < DISCOVERY_SHARE + COLLABORATIVE_SHARE and peers:
            row = fetch_collaborative_data(user_id, random.choice(peers))
        if not take(row):
            # interest-weighted draw; if that genre is exhausted, try the others by weight
            order = random.choices(genres, weights=[weights[g] for g in genres], k=len(genres))
            for genre in dict.fromkeys(order):
                if take(_pick_interest(user_id, genre)):
                    break
            else:
                break                                # nothing unseen left anywhere

    cards = _cards(user_id, picked)
    if len(cards) < limit:
        cards += _cards(user_id, _recycle(user_id, picked_ids, limit - len(cards)))
    return cards


def search(user_id, query, limit=30):
    """Caption search over Active items (does not count as a view)."""
    ids = vault.search_ids(query)
    if not ids:
        return []
    marks = ", ".join(["%s"] * len(ids))
    with db_session() as (db, cursor):
        cursor.execute(f"SELECT id, genre, filepath FROM ranker WHERE status = 'Active' AND id IN ({marks}) "
                       f"ORDER BY (likes - dislikes) DESC LIMIT %s", (*ids, limit))
        rows = cursor.fetchall()
    return _cards(user_id, rows)


def get_saved(user_id):
    saved = Settings.get_saved_blogs(user_id)
    return [{
        "id": s["blog_id"], "niche": s["niche"], "title": s["title"], "description": s["description"],
        "image_url": s["image_url"], "liked": bool(s["is_liked"]), "disliked": bool(s["is_disliked"]),
        "saved": True, "pending": s["niche"] == NOISE,
    } for s in saved]


# -------------------------------------------------------------
# INTERACTIONS
# -------------------------------------------------------------
def interact(user_id, blog_id, action):
    """
    Toggle like / dislike / save on a post. Returns (ok, state_or_message).
    like and dislike are mutually exclusive; each moves the user's interest in
    that genre and the post's ranker counters.
    """
    if action not in INTEREST_POINTS:
        return False, "Unknown action."

    with db_session() as (db, cursor):
        cursor.execute("SELECT genre, status FROM ranker WHERE id = %s AND status IN ('Active', %s)",
                       (blog_id, PENDING))
        row = cursor.fetchone()
        if not row:
            return False, "That post no longer exists."
        genre, status = row
        if status == PENDING and action != "save":
            return False, "This post is still waiting for a niche, so it can't be rated yet."
        _, liked, disliked, saved, created = _get_or_create_history(cursor, user_id, blog_id)
        if created:
            cursor.execute("UPDATE ranker SET views = views + 1 WHERE id = %s", (blog_id,))

    if action == "save" and saved:                   # un-saving has its own tested helper
        ok, message = Settings.remove_saved_blog(user_id, blog_id, genre)
        return (True, {"liked": liked, "disliked": disliked, "saved": False}) if ok else (False, message)

    with db_session() as (db, cursor):
        if action == "like":
            if liked:
                liked = False
                cursor.execute("UPDATE ranker SET likes = GREATEST(0, likes - 1) WHERE id = %s", (blog_id,))
                _bump_interest(cursor, user_id, genre, -INTEREST_POINTS["like"])
            else:
                liked = True
                cursor.execute("UPDATE ranker SET likes = likes + 1 WHERE id = %s", (blog_id,))
                _bump_interest(cursor, user_id, genre, INTEREST_POINTS["like"])
                if disliked:                         # switching sides undoes the dislike
                    disliked = False
                    cursor.execute("UPDATE ranker SET dislikes = GREATEST(0, dislikes - 1) WHERE id = %s", (blog_id,))
                    _bump_interest(cursor, user_id, genre, -INTEREST_POINTS["dislike"])
        elif action == "dislike":
            if disliked:
                disliked = False
                cursor.execute("UPDATE ranker SET dislikes = GREATEST(0, dislikes - 1) WHERE id = %s", (blog_id,))
                _bump_interest(cursor, user_id, genre, -INTEREST_POINTS["dislike"])
            else:
                disliked = True
                cursor.execute("UPDATE ranker SET dislikes = dislikes + 1 WHERE id = %s", (blog_id,))
                _bump_interest(cursor, user_id, genre, INTEREST_POINTS["dislike"])
                if liked:
                    liked = False
                    cursor.execute("UPDATE ranker SET likes = GREATEST(0, likes - 1) WHERE id = %s", (blog_id,))
                    _bump_interest(cursor, user_id, genre, -INTEREST_POINTS["like"])
        else:  # save (the un-save case returned above)
            saved = True
            cursor.execute("UPDATE ranker SET saves = saves + 1 WHERE id = %s", (blog_id,))
            if genre != NOISE:                       # "Noise" is not a taste the user has
                _bump_interest(cursor, user_id, genre, INTEREST_POINTS["save"])

        cursor.execute("""
            UPDATE user_history SET is_liked = %s, is_disliked = %s, is_saved = %s, interacted_at = NOW()
            WHERE user_id = %s AND blog_id = %s
        """, (int(liked), int(disliked), int(saved), user_id, blog_id))

    if action in ("like", "dislike"):
        refresh_peers(user_id)
    return True, {"liked": liked, "disliked": disliked, "saved": saved}


def refresh_peers(user_id):
    """Peers = the users who liked the most of the same posts as this user."""
    with db_session() as (db, cursor):
        cursor.execute("""
            SELECT h2.user_id, COUNT(*) AS common
            FROM user_history h1
            JOIN user_history h2 ON h1.blog_id = h2.blog_id AND h2.user_id <> h1.user_id
            WHERE h1.user_id = %s AND h1.is_liked = 1 AND h2.is_liked = 1
            GROUP BY h2.user_id ORDER BY common DESC LIMIT %s
        """, (user_id, PEER_LIMIT))
        peers = cursor.fetchall()
        cursor.execute("DELETE FROM collaborative_map WHERE user_id = %s", (user_id,))
        for peer_id, common in peers:
            cursor.execute("INSERT INTO collaborative_map (user_id, peer_id, similarity_score) VALUES (%s, %s, %s)",
                           (user_id, peer_id, common))


# -------------------------------------------------------------
# UPLOADS
# -------------------------------------------------------------
def register_upload(user_id, asset_id, niche):
    """
    Add a freshly stored upload to ranker and to the uploader's Saved list.
    A confident niche goes live at once; NOISE goes in as Pending (visible only to
    its uploader) until discover.py groups it into a new niche.
    """
    status = PENDING if niche == NOISE else "Active"
    with db_session() as (db, cursor):
        cursor.execute(
            "INSERT INTO ranker (id, filepath, genre, status, saves) VALUES (%s, %s, %s, %s, 1)",
            (asset_id, vault.relative_filepath(asset_id), niche, status))
        cursor.execute(
            "INSERT INTO user_history (history_id, user_id, blog_id, is_saved) VALUES (%s, %s, %s, 1)",
            (generate_id(10), user_id, asset_id))
        if niche != NOISE:
            _bump_interest(cursor, user_id, niche, INTEREST_POINTS["save"])


def asset_id_taken(asset_id):
    with db_session() as (db, cursor):
        cursor.execute("SELECT 1 FROM ranker WHERE id = %s", (asset_id,))
        taken = cursor.fetchone() is not None
    return taken or os.path.exists(vault.image_path(asset_id)) or os.path.exists(vault.caption_path(asset_id))


def new_asset_id():
    """A 7-character id (same style as the existing vault) that no row or file uses yet."""
    for _ in range(20):
        candidate = generate_id(7)
        if not asset_id_taken(candidate):
            return candidate
    raise RuntimeError("Couldn't find a free asset id.")

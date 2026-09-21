import os
import mysql.connector
import secrets
import string
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )

@contextmanager
def db_session(dictionary=False):
    """Yield (db, cursor). Commits on success, rolls back on error, always closes."""
    db = get_connection()
    cursor = db.cursor(dictionary=dictionary)
    try:
        yield db, cursor
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        cursor.close()
        db.close()

def generate_id(length=7):
    chars = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(chars) for _ in range(length))

def fetch_discovery_data(user_id, exclude_top=3):
    """
    One unseen Active item from OUTSIDE the user's `exclude_top` strongest
    genres. The default of 3 leaves nothing to discover when there are only
    3 niches, so callers with few genres should pass a smaller number.
    """
    db = get_connection()
    cursor = db.cursor()
    cursor.execute("SELECT genre FROM user_interests WHERE user_id = %s ORDER BY points DESC LIMIT %s", (user_id, int(exclude_top)))
    exclude = [row[0] for row in cursor.fetchall()] or ['_NONE_']
    placeholders = ', '.join(['%s'] * len(exclude))
    query = f"""
        SELECT r.id, r.genre, r.filepath FROM ranker r
        LEFT JOIN user_history h ON r.id = h.blog_id AND h.user_id = %s
        WHERE r.genre NOT IN ({placeholders}) AND r.status = 'Active' AND h.blog_id IS NULL
        ORDER BY (r.likes - r.dislikes) DESC, RAND() LIMIT 1
    """
    cursor.execute(query, (user_id, *exclude))
    result = cursor.fetchone()
    cursor.close()
    db.close()
    return result  # Returns (id, genre, filepath) or None

def fetch_collaborative_data(user_id, peer_id):
    db = get_connection()
    cursor = db.cursor()
    query = """
        SELECT r.id, r.genre, r.filepath FROM user_history h
        JOIN ranker r ON h.blog_id = r.id
        LEFT JOIN user_history my_h ON r.id = my_h.blog_id AND my_h.user_id = %s
        WHERE h.user_id = %s AND h.is_liked = 1 AND my_h.blog_id IS NULL AND r.status = 'Active'
        ORDER BY RAND() LIMIT 1
    """
    cursor.execute(query, (user_id, peer_id))
    result = cursor.fetchone()
    cursor.close()
    db.close()
    return result  # Returns (id, genre, filepath) or None
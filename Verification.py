"""
Accounts: sign up / sign in, password checks, account edits, deletion and
"forgot password". Every function opens its own short-lived DB connection
(no module-level connection) and returns plain values, so the Flask routes
in app.py stay thin.
"""
import re
from datetime import datetime

import bcrypt
import mysql.connector

import Settings
from Connector import db_session, generate_id

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.]{3,30}$")
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_BYTES = 72          # bcrypt silently ignores anything past 72 bytes
DISPLAY_NAME_MAX = 100

# Compared against when a username doesn't exist, so "no such user" and
# "wrong password" take the same time and can't be told apart.
_DUMMY_HASH = bcrypt.hashpw(b"neuroed-dummy-password", bcrypt.gensalt())


def _hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()


def _password_problem(password):
    if password is None or len(password) < MIN_PASSWORD_LEN:
        return f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return "Password is too long (72 bytes max)."
    return None


def _username_problem(username):
    if not username or not USERNAME_RE.match(username):
        return "Username must be 3-30 characters: letters, numbers, '_' or '.'."
    return None


# -------------------------------------------------------------
# SIGN UP / SIGN IN
# -------------------------------------------------------------
def sign_up(username, password, display_name):
    """Create an account. Returns (ok, message)."""
    username = (username or "").strip()
    display_name = " ".join((display_name or "").split())

    problem = _username_problem(username) or _password_problem(password)
    if problem:
        return False, problem
    if not display_name or len(display_name) > DISPLAY_NAME_MAX:
        return False, f"Display name is required (max {DISPLAY_NAME_MAX} characters)."

    hashed = _hash_password(password)
    for _ in range(5):                       # retry only on a user_id collision
        user_id = generate_id(length=7)
        try:
            with db_session() as (db, cursor):
                cursor.execute(
                    "INSERT INTO users (user_id, username, display_name, password_hash, status) "
                    "VALUES (%s, %s, %s, %s, 'active')",
                    (user_id, username, display_name, hashed)
                )
            return True, f"Welcome, {display_name}! Please sign in."
        except mysql.connector.errors.IntegrityError as err:
            if "username" in str(err).lower():
                return False, "That username is already taken."
            continue
    return False, "Couldn't create the account. Please try again."


def verify_user(username, password):
    """Check credentials. Returns the user_id on success, else None."""
    username = (username or "").strip()
    with db_session() as (db, cursor):
        cursor.execute(
            "SELECT user_id, password_hash FROM users WHERE username = %s AND status = 'active'",
            (username,)
        )
        row = cursor.fetchone()

    stored = row[1].encode("utf-8") if row else _DUMMY_HASH
    try:
        matches = bcrypt.checkpw((password or "").encode("utf-8"), stored)
    except ValueError:
        matches = False
    if not row or not matches:
        return None

    apply_decay(row[0])
    return row[0]


def apply_decay(user_id):
    """Interests fade 1% per day since the last update."""
    with db_session() as (db, cursor):
        cursor.execute("SELECT MAX(last_updated) FROM user_interests WHERE user_id = %s", (user_id,))
        last_log = cursor.fetchone()[0]
        if last_log:
            days_passed = (datetime.now() - last_log).days
            if days_passed > 0:
                cursor.execute("""
                    UPDATE user_interests
                    SET points = ROUND(points * %s, 5), last_updated = NOW()
                    WHERE user_id = %s
                """, (pow(0.99, days_passed), user_id))


# -------------------------------------------------------------
# ACCOUNT DETAILS
# -------------------------------------------------------------
def get_account(user_id):
    """Profile + recovery-email status for the dashboard, or None if the user is gone."""
    with db_session(dictionary=True) as (db, cursor):
        cursor.execute("""
            SELECT u.username, u.display_name, s.email, COALESCE(s.is_verified, 0) AS email_verified
            FROM users u LEFT JOIN safety s ON s.user_id = u.user_id
            WHERE u.user_id = %s AND u.status = 'active'
        """, (user_id,))
        row = cursor.fetchone()
    if not row:
        return None
    return {
        "username": row["username"],
        "display_name": row["display_name"],
        "email": row["email"] or "",
        "email_verified": bool(row["email_verified"]),
    }


def check_password(user_id, password):
    with db_session() as (db, cursor):
        cursor.execute("SELECT password_hash FROM users WHERE user_id = %s AND status = 'active'", (user_id,))
        row = cursor.fetchone()
    if not row:
        return False
    try:
        return bcrypt.checkpw((password or "").encode("utf-8"), row[0].encode("utf-8"))
    except ValueError:
        return False


def update_account(user_id, current_password, username=None, display_name=None, new_password=None):
    """
    Change username / display name / password. The current password is
    re-checked here on every call - the dashboard's "unlock" step is only a
    convenience, never the security boundary. Returns (ok, message).
    """
    if not check_password(user_id, current_password):
        return False, "Incorrect password."

    fields, values = [], []
    if username is not None:
        username = username.strip()
        problem = _username_problem(username)
        if problem:
            return False, problem
        fields.append("username = %s"); values.append(username)
    if display_name is not None:
        display_name = " ".join(display_name.split())
        if not display_name or len(display_name) > DISPLAY_NAME_MAX:
            return False, f"Display name is required (max {DISPLAY_NAME_MAX} characters)."
        fields.append("display_name = %s"); values.append(display_name)
    if new_password:
        problem = _password_problem(new_password)
        if problem:
            return False, problem
        fields.append("password_hash = %s"); values.append(_hash_password(new_password))

    if not fields:
        return True, "Nothing to change."
    try:
        with db_session() as (db, cursor):
            cursor.execute(f"UPDATE users SET {', '.join(fields)} WHERE user_id = %s", (*values, user_id))
    except mysql.connector.errors.IntegrityError:
        return False, "That username is already taken."
    return True, "Settings updated."


def delete_account(user_id, password):
    """
    Permanently delete the account after re-checking the password. Foreign keys
    cascade to safety, user_history, user_interests and collaborative_map.
    Returns (ok, message).
    """
    if not check_password(user_id, password):
        return False, "Incorrect password."
    with db_session() as (db, cursor):
        cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
    return True, "Account deleted."


# -------------------------------------------------------------
# FORGOT PASSWORD (email code)
# -------------------------------------------------------------
def start_password_reset(username):
    """
    Email a code if the account exists AND has a verified recovery email.
    Deliberately returns nothing: the route always answers the same way, so
    this can't be used to find out which usernames exist.
    """
    username = (username or "").strip()
    with db_session() as (db, cursor):
        cursor.execute("SELECT user_id FROM users WHERE username = %s AND status = 'active'", (username,))
        row = cursor.fetchone()
    if row:
        Settings.send_reset_otp(row[0])


def finish_password_reset(username, code, new_password):
    """Set a new password if `code` is right. Returns (ok, message)."""
    problem = _password_problem(new_password)
    if problem:
        return False, problem

    username = (username or "").strip()
    with db_session() as (db, cursor):
        cursor.execute("SELECT user_id FROM users WHERE username = %s AND status = 'active'", (username,))
        row = cursor.fetchone()
    if not row:
        return False, "That code is invalid or has expired."

    ok, message = Settings.check_otp(row[0], code)
    if not ok:
        return False, message

    with db_session() as (db, cursor):
        cursor.execute("UPDATE users SET password_hash = %s WHERE user_id = %s",
                       (_hash_password(new_password), row[0]))
    return True, "Password updated. Please sign in."

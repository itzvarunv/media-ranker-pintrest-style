"""
Account safety (email + one-time codes) and saved items.

The recovery email lives in the `safety` table. A code is only ever emailed to
an address the user has already proven they own (`is_verified = 1`), which is
what makes "forgot password" safe.
"""
import os
import re
import secrets
import smtplib
import string
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import bcrypt
import mysql.connector

import vault
from Connector import db_session

OTP_LENGTH = 6
OTP_TTL_MINUTES = 10
OTP_MAX_ATTEMPTS = 5          # wrong guesses allowed per issued code
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Wrong guesses against the CURRENT code, per user (in memory). After
# OTP_MAX_ATTEMPTS the code is wiped from the database, so it can't be brute-forced.
_failed_attempts = {}


# -------------------------------------------------------------
# EMAIL DELIVERY
# -------------------------------------------------------------
def send_otp_email(target_email, otp_code):
    sender_email = os.getenv("MAIL_USERNAME")
    sender_password = os.getenv("MAIL_PASSWORD")

    msg = MIMEMultipart()
    msg['From'] = f"Neuroed Support <{sender_email}>"
    msg['To'] = target_email
    msg['Subject'] = "Your Neuroed Verification Code"
    body = (f"Your verification code is: {otp_code}\n\n"
            f"This code expires in {OTP_TTL_MINUTES} minutes. "
            f"If you didn't request it, you can ignore this email.")
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(os.getenv("MAIL_SERVER", "smtp.gmail.com"),
                              int(os.getenv("MAIL_PORT", 587)), timeout=15)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


# -------------------------------------------------------------
# ONE-TIME CODES
# -------------------------------------------------------------
def _new_otp():
    return ''.join(secrets.choice(string.digits) for _ in range(OTP_LENGTH))


def _hash_otp(otp):
    return bcrypt.hashpw(otp.encode('utf-8'), bcrypt.gensalt()).decode()


def manage_safety(user_id, email):
    """
    Set (or change) the recovery email and email a verification code to it.
    The address stays unverified - and unusable for password recovery - until
    the code is confirmed with confirm_email().
    """
    email = (email or "").strip()
    if len(email) > 255 or not EMAIL_RE.match(email):
        return False, "Please enter a valid email address."

    otp = _new_otp()
    hashed, expires = _hash_otp(otp), datetime.now() + timedelta(minutes=OTP_TTL_MINUTES)
    try:
        with db_session() as (db, cursor):
            # Explicit select + update/insert, NOT "INSERT ... ON DUPLICATE KEY UPDATE":
            # `safety` also has a UNIQUE email, so that clause would fire when the
            # email belongs to a DIFFERENT user and silently overwrite THEIR row.
            cursor.execute("SELECT email FROM safety WHERE user_id = %s", (user_id,))
            row = cursor.fetchone()
            if row:
                same_address = row[0].lower() == email.lower()
                cursor.execute("""
                    UPDATE safety
                    SET email = %s, otp_hash = %s, otp_expires_at = %s,
                        is_verified = IF(%s, is_verified, 0)
                    WHERE user_id = %s
                """, (email, hashed, expires, same_address, user_id))
            else:
                cursor.execute("""
                    INSERT INTO safety (user_id, email, otp_hash, otp_expires_at, is_verified)
                    VALUES (%s, %s, %s, %s, 0)
                """, (user_id, email, hashed, expires))
    except mysql.connector.errors.IntegrityError:
        return False, "That email address is already in use."

    _failed_attempts.pop(user_id, None)
    if send_otp_email(email, otp):
        return True, f"We sent a {OTP_LENGTH}-digit code to {email}."
    return False, "We couldn't send the email. Check the mail settings and try again."


def send_reset_otp(user_id):
    """Email a fresh code to the user's VERIFIED recovery email. False if they have none."""
    otp = _new_otp()
    with db_session() as (db, cursor):
        cursor.execute("SELECT email FROM safety WHERE user_id = %s AND is_verified = 1", (user_id,))
        row = cursor.fetchone()
        if not row or not row[0]:
            return False
        cursor.execute(
            "UPDATE safety SET otp_hash = %s, otp_expires_at = %s WHERE user_id = %s",
            (_hash_otp(otp), datetime.now() + timedelta(minutes=OTP_TTL_MINUTES), user_id)
        )
        email = row[0]
    _failed_attempts.pop(user_id, None)
    return send_otp_email(email, otp)


def check_otp(user_id, code):
    """Validate (and on success consume) the user's current code. Returns (ok, message)."""
    invalid = (False, "That code is invalid or has expired.")
    code = (code or "").strip()

    with db_session() as (db, cursor):
        cursor.execute("SELECT otp_hash, otp_expires_at FROM safety WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if not row or not row[0] or not row[1] or datetime.now() > row[1]:
            return invalid

        good = (code.isdigit() and len(code) == OTP_LENGTH
                and bcrypt.checkpw(code.encode('utf-8'), row[0].encode('utf-8')))
        if not good:
            _failed_attempts[user_id] = _failed_attempts.get(user_id, 0) + 1
            if _failed_attempts[user_id] >= OTP_MAX_ATTEMPTS:
                cursor.execute("UPDATE safety SET otp_hash = NULL, otp_expires_at = NULL WHERE user_id = %s", (user_id,))
                return False, "Too many wrong codes. Please request a new one."
            return invalid

        cursor.execute("UPDATE safety SET otp_hash = NULL, otp_expires_at = NULL WHERE user_id = %s", (user_id,))
    _failed_attempts.pop(user_id, None)
    return True, "OK"


def confirm_email(user_id, code):
    """Verify the code sent by manage_safety() and mark the recovery email as verified."""
    ok, message = check_otp(user_id, code)
    if not ok:
        return False, message
    with db_session() as (db, cursor):
        cursor.execute("UPDATE safety SET is_verified = 1 WHERE user_id = %s", (user_id,))
    return True, "Recovery email verified."


# -------------------------------------------------------------
# SAVED ITEMS
# -------------------------------------------------------------
def get_saved_blogs(user_id):
    """Saved image blogs for the user, newest first, with their vault caption."""
    with db_session(dictionary=True) as (db, cursor):
        cursor.execute("""
            SELECT h.blog_id, r.genre AS niche, r.filepath, h.is_liked, h.is_disliked
            FROM user_history h
            JOIN ranker r ON h.blog_id = r.id
            WHERE h.user_id = %s AND h.is_saved = 1 AND r.status IN ('Active', 'Pending')
            ORDER BY h.interacted_at DESC
        """, (user_id,))
        saves = cursor.fetchall()

    for save in saves:
        caption = vault.read_caption(save['blog_id'])
        save['filename'] = os.path.basename(save['filepath'])
        save['title'] = caption['title']
        save['description'] = caption['description']
        save['image_url'] = f"/vault/{save['filename']}"
    return saves


def remove_saved_blog(user_id, blog_id, genre):
    """Unsaves an image blog entry and updates interest rankings."""
    try:
        with db_session() as (db, cursor):   # any DB error rolls the whole unsave back
            cursor.execute("""
                UPDATE user_history SET is_saved = 0, interacted_at = NOW()
                WHERE user_id = %s AND blog_id = %s AND is_saved = 1
            """, (user_id, blog_id))
            if cursor.rowcount == 0:
                return True, "Already removed."     # nothing was saved: leave counters alone

            cursor.execute("UPDATE ranker SET saves = GREATEST(0, saves - 1) WHERE id = %s", (blog_id,))
            cursor.execute("""
                UPDATE user_interests SET points = GREATEST(0, points - 5)
                WHERE user_id = %s AND genre = %s
            """, (user_id, genre))
        return True, "Removed from saved items!"
    except mysql.connector.Error as err:
        return False, f"Database error: {err}"

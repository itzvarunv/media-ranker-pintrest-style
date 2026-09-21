import os
import smtplib
import random
import bcrypt
import hashlib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from Connector import get_connection

def send_otp_email(target_email, otp_code):
    sender_email = os.getenv("MAIL_USERNAME")
    sender_password = os.getenv("MAIL_PASSWORD")
    
    msg = MIMEMultipart()
    msg['From'] = f"Neuroed Support <{sender_email}>"
    msg['To'] = target_email
    msg['Subject'] = "Your Neuroed Verification Code"
    
    body = f"Your verification code is: {otp_code}\n\nThis code expires in 10 minutes."
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP(os.getenv("MAIL_SERVER", "smtp.gmail.com"), int(os.getenv("MAIL_PORT", 587)))
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def manage_safety(user_id, email):
    """Generates a 4-digit OTP, stores its hash in `safety`, and emails it to the user."""
    db = get_connection()
    cursor = db.cursor()
    try:
        # 1. Generate 4-digit OTP
        otp = str(random.randint(1000, 9999))
        otp_hash = bcrypt.hashpw(otp.encode('utf-8'), bcrypt.gensalt()).decode()
        expires_at = datetime.now() + timedelta(minutes=10)

        # 2. Update or Insert into `safety` table
        cursor.execute("""
            INSERT INTO safety (user_id, email, otp_hash, otp_expires_at, is_verified)
            VALUES (%s, %s, %s, %s, 0)
            ON DUPLICATE KEY UPDATE 
                email = VALUES(email),
                otp_hash = VALUES(otp_hash),
                otp_expires_at = VALUES(otp_expires_at)
        """, (user_id, email, otp_hash, expires_at))
        
        db.commit()

        # 3. Send email from MAIL_USERNAME to the user's email
        if send_otp_email(email, otp):
            return True, "OTP sent successfully!"
        return False, "Failed to send email."
    finally:
        cursor.close()
        db.close()

VAULT_DIR = os.path.join(os.path.dirname(__file__), 'vault')

def get_saved_blogs(user_id):
    """Fetches saved image blogs for the user, classified by niche/genre in SQL."""
    db = get_connection()
    cursor = db.cursor(dictionary=True)
    try:
        query = """
            SELECT h.blog_id, r.genre AS niche, r.filepath 
            FROM user_history h
            JOIN ranker r ON h.blog_id = r.id
            WHERE h.user_id = %s AND h.is_saved = 1
        """
        cursor.execute(query, (user_id,))
        saves = cursor.fetchall()
        
        # Attach the image filename and formatted title
        for save in saves:
            filename = os.path.basename(save['filepath'])
            save['filename'] = filename
            save['title'] = os.path.splitext(filename)[0].replace('_', ' ').title()
            save['image_url'] = f"/vault/{filename}"
            
        return saves
    finally:
        cursor.close()
        db.close()

def remove_saved_blog(user_id, blog_id, genre):
    """Unsaves an image blog entry and updates interest rankings."""
    db = get_connection()
    cursor = db.cursor()
    try:
        cursor.execute("""
            UPDATE user_history SET is_saved = 0 
            WHERE user_id = %s AND blog_id = %s
        """, (user_id, blog_id))
        
        cursor.execute("""
            UPDATE ranker SET saves = GREATEST(0, saves - 1) 
            WHERE id = %s
        """, (blog_id,))
        
        cursor.execute("""
            UPDATE user_interests 
            SET points = GREATEST(0, points - 5) 
            WHERE user_id = %s AND genre = %s
        """, (user_id, genre))
        
        db.commit()
        return True, "Removed from saved items!"
    except mysql.connector.Error as err:
        db.rollback()
        return False, f"Database error: {err}"
    finally:
        cursor.close()
        db.close()

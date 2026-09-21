import bcrypt
import mysql.connector
import secrets
import string
import hashlib
from datetime import datetime, timedelta
from Connector import get_connection, generate_id

db = get_connection()
cursor = db.cursor()

def SignUp():
    print("Creating your new account!")
    display_name = input("What should we call you? (Display Name): ")
    while True:
        username = input("Pick a unique Username for login: ")
        cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            print("That username is already taken! Please try a different one.")
        else:
            break 
    password = input("Enter a password: ")
    password_bytes = password.encode("utf-8")
    hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode()
    while True:
        try:
            u_id = generate_id(length=7) 
            cursor.execute(
                "INSERT INTO users (user_id, username, display_name, password_hash, status) VALUES (%s, %s, %s, %s, 'active')",
                (u_id, username, display_name, hashed)
            )
            db.commit()
            print(f"Success! Welcome {display_name}. You can now login using your username: {username}")
            return True, u_id 
        except mysql.connector.errors.IntegrityError:
            continue

def SignIn():
    username = input("What's your username? ")
    cursor.execute(
        "SELECT user_id, password_hash FROM users WHERE username = %s AND status = 'active'",
        (username,)
    )
    data = cursor.fetchone() 
    if data is None:
        print("User doesn't exist or account is deactivated!")
        return False, None
    stored_hash = data[1].encode('utf-8') 
    password = input("Enter password: ")
    if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
        print("Login successful!")
        confirm = True
        user_id = data[0]
        apply_decay(user_id)
    else:
        print("Wrong password!")
        confirm = False
        user_id = None     
    return confirm, user_id

def forgot_password():
    username = input("Enter your username: ")
    cursor.execute("SELECT user_id FROM users WHERE username = %s AND status = 'active'", (username,))
    user_data = cursor.fetchone()    
    if not user_data:
        print("Account not found.")
        return
        
    u_id = user_data[0]
    cursor.execute("SELECT email FROM safety WHERE user_id = %s", (u_id,))
    safety_data = cursor.fetchone()
    
    if not safety_data or not safety_data[0]:
        print("No recovery email linked to this account.")
        return

    email = safety_data[0]
    
    # Generate 4-digit OTP
    raw_otp = ''.join(secrets.choice(string.digits) for _ in range(4))
    hashed_otp = hash_function(raw_otp)
    expires_at = datetime.now() + timedelta(minutes=10)
    
    # Store/Update OTP in database
    cursor.execute("""
        UPDATE safety 
        SET otp_hash = %s, otp_expires_at = %s 
        WHERE user_id = %s
    """, (hashed_otp, expires_at, u_id))
    db.commit()
    
    # Send verification code via simulated email dispatch
    send_otp_email(email, raw_otp)
    
    # Verify OTP input
    user_otp = input("\nEnter the 4-digit verification code sent to your email: ").strip()
    
    if datetime.now() > expires_at:
        print(" Verification code has expired. Please try again.")
        return
        
    if hash_function(user_otp) == hashed_otp:
        print(" Identity Confirmed!")
        new_pass = input("Enter NEW password: ")
        new_hashed = bcrypt.hashpw(new_pass.encode(), bcrypt.gensalt()).decode()
        
        # Reset password and invalidate OTP
        cursor.execute("UPDATE users SET password_hash = %s WHERE user_id = %s", (new_hashed, u_id))
        cursor.execute("UPDATE safety SET otp_hash = NULL, otp_expires_at = NULL WHERE user_id = %s", (u_id,))
        db.commit()
        print(" Password updated successfully!")
    else:
        print(" Incorrect verification code.")

def send_otp_email(target_email, otp_code):
    print(f"\n[DEV SIMULATION] Email sent to {target_email} with OTP: {otp_code}")

def apply_decay(user_id):
    cursor.execute("SELECT MAX(last_updated) FROM user_interests WHERE user_id = %s", (user_id,))
    last_log = cursor.fetchone()[0]
    if last_log:
        now = datetime.now()
        days_passed = (now - last_log).days
        if days_passed > 0:
            decay_factor = pow(0.99, days_passed)
            cursor.execute("""
                UPDATE user_interests 
                SET points = ROUND(points * %s, 5), 
                    last_updated = NOW() 
                WHERE user_id = %s
            """, (decay_factor, user_id))
            db.commit()

def hash_function(text):
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()
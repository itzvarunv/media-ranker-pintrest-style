import os
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

sender_email = os.getenv("MAIL_USERNAME")
sender_password = os.getenv("MAIL_PASSWORD")
target_email = input()

# Create the email message
msg = MIMEText("yes you did it")
msg['From'] = f"Neuroed Support <{sender_email}>"
msg['To'] = target_email
msg['Subject'] = "Neuroed SMTP Test"

try:
    # Connect to Gmail SMTP server
    server = smtplib.SMTP(os.getenv("MAIL_SERVER", "smtp.gmail.com"), int(os.getenv("MAIL_PORT", 587)))
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(msg)
    server.quit()
    print("Email sent successfully!")
except Exception as e:
    print(f"Error sending email: {e}")

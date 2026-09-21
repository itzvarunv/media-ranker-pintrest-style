from flask import Flask, render_template, request, flash, redirect, url_for, send_from_directory
import Verification

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Needed for flash messages

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/vault/<filename>')
def serve_vault_image(filename):
    return send_from_directory('vault', filename)

@app.route('/auth', methods=['POST'])
def handle_auth():
    display_name = request.form.get('display_name')
    username = request.form.get('username')
    password = request.form.get('password')

    if display_name:
        print(f"Attempting Sign Up for: {username}")
        success = Verification.sign_up(username, password, display_name) 
        if success:
            flash("Sign up successful! Please log in.", "success")
            return redirect(url_for('home'))
        else:
            flash("Sign up failed.", "danger")
            return redirect(url_for('home'))
    else:
        print(f"Attempting Sign In for: {username}")
        success = Verification.verify_user(username, password) 
        if success:
            return render_template('dashboard.html')
        else:
            flash("Invalid credentials.", "danger")
            return redirect(url_for('home'))
        
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8080)

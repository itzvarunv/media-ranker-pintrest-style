import os
import secrets
from functools import wraps

from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)

import Settings
import Verification
import classifier
import feed
import vault

app = Flask(__name__)

# Sessions are signed with this key. Put SECRET_KEY=<long random string> in .env
# to stay signed in across restarts; without it a fresh key is made every start.
app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(32)
if not os.getenv("SECRET_KEY"):
    print("[i] SECRET_KEY not set in .env - sessions will reset whenever the app restarts.")

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",          # also blocks cross-site form/JSON POSTs (CSRF)
    MAX_CONTENT_LENGTH=vault.MAX_UPLOAD_BYTES + 512 * 1024,
)


@app.after_request
def security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    return response


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------
def current_user_id():
    return session.get("user_id")


def login_required(view):
    """Pages redirect to the login screen; /api/* answers 401 JSON."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user_id():
            if request.path.startswith("/api/"):
                return jsonify(ok=False, message="Please sign in again."), 401
            return redirect(url_for("home"))
        return view(*args, **kwargs)
    return wrapped


def body():
    return request.get_json(silent=True) or {}


def fail(message, status=400, **extra):
    return jsonify(ok=False, message=message, **extra), status


# -------------------------------------------------------------
# PAGES + AUTH
# -------------------------------------------------------------
@app.route("/")
def home():
    if current_user_id():
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/auth", methods=["POST"])
def handle_auth():
    username = request.form.get("username", "")
    password = request.form.get("password", "")

    if request.form.get("mode") == "signup":
        ok, message = Verification.sign_up(username, password, request.form.get("display_name", ""))
        flash(message, "success" if ok else "danger")
        return redirect(url_for("home"))

    user_id = Verification.verify_user(username, password)
    if not user_id:
        flash("Invalid username or password.", "danger")
        return redirect(url_for("home"))

    session.clear()                          # new session on login (no fixation)
    session["user_id"] = user_id
    return redirect(url_for("dashboard"))


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify(ok=True)


@app.route("/dashboard")
@login_required
def dashboard():
    account = Verification.get_account(current_user_id())
    if not account:                          # account deleted elsewhere
        session.clear()
        return redirect(url_for("home"))
    return render_template("dashboard.html", account=account,
                           max_upload_mb=vault.MAX_UPLOAD_BYTES // (1024 * 1024))


@app.route("/vault/<path:filename>")
@login_required
def serve_vault_image(filename):
    # Images only: the same folder holds the caption .txt files.
    if os.path.splitext(filename)[1].lower() not in vault.SERVE_EXTS:
        return fail("Not found.", 404)
    return send_from_directory(vault.VAULT_DIR, filename)


# -------------------------------------------------------------
# FORGOT PASSWORD
# -------------------------------------------------------------
@app.route("/forgot-password")
def forgot_password():
    return render_template("forgot.html", otp_length=Settings.OTP_LENGTH)


@app.route("/forgot-password/request", methods=["POST"])
def forgot_password_request():
    Verification.start_password_reset(body().get("username", ""))
    # Same answer whether or not the account exists / has a verified email.
    return jsonify(ok=True, message="If that account has a verified recovery email, we've sent it a code.")


@app.route("/forgot-password/reset", methods=["POST"])
def forgot_password_reset():
    data = body()
    ok, message = Verification.finish_password_reset(
        data.get("username", ""), data.get("otp", ""), data.get("new_password", ""))
    return (jsonify(ok=True, message=message) if ok else fail(message))


# -------------------------------------------------------------
# FEED / SAVED / INTERACTIONS
# -------------------------------------------------------------
@app.route("/api/feed")
@login_required
def api_feed():
    query = request.args.get("q", "").strip()
    limit = min(max(request.args.get("limit", 12, type=int), 1), 30)
    items = feed.search(current_user_id(), query, limit) if query else feed.get_feed(current_user_id(), limit)
    return jsonify(ok=True, items=items, query=query)


@app.route("/api/saved")
@login_required
def api_saved():
    return jsonify(ok=True, items=feed.get_saved(current_user_id()))


@app.route("/api/interact", methods=["POST"])
@login_required
def api_interact():
    data = body()
    ok, result = feed.interact(current_user_id(), str(data.get("blog_id", "")), data.get("action", ""))
    return jsonify(ok=True, **result) if ok else fail(result)


# -------------------------------------------------------------
# UPLOAD
# -------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
@login_required
def api_upload():
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return fail("Choose an image first.")
    stem, ext = os.path.splitext(upload.filename)
    if ext.lower() not in vault.UPLOAD_EXTS:
        return fail("Only JPG, PNG and WEBP images are supported.")

    data = upload.read(vault.MAX_UPLOAD_BYTES + 1)
    if len(data) > vault.MAX_UPLOAD_BYTES:
        return fail(f"That image is larger than {vault.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)

    title = request.form.get("title", "").strip() or stem
    description = request.form.get("description", "")

    asset_id = feed.new_asset_id()
    try:
        path = vault.save_image(data, asset_id)
    except ValueError as err:
        return fail(str(err))

    try:
        result = classifier.classify(path)
    except classifier.ClassifierUnavailable as err:
        vault.delete_asset_files(asset_id)
        return fail(f"The classifier isn't ready: {err}", 503)

    # "Not too confident" -> Noise: kept (only its uploader sees it) until
    # discover.py groups enough similar Noise images into a new niche.
    pending = result["rejected"]
    niche = feed.NOISE if pending else result["niche"]
    try:
        title, description = vault.write_caption(asset_id, niche, title, description)
        feed.register_upload(current_user_id(), asset_id, niche)
    except Exception:
        vault.delete_asset_files(asset_id)   # never leave an orphaned file behind
        raise

    return jsonify(ok=True, niche=niche, pending=pending, confidence=result["confidence"],
                   best_guess=result["best_niche"] if pending else None, scores=result["scores"],
                   item={"id": asset_id, "niche": niche, "title": title, "description": description,
                         "image_url": f"/vault/{asset_id}.jpeg", "liked": False, "disliked": False,
                         "saved": True, "pending": pending})


@app.errorhandler(413)
def too_large(_err):
    return fail(f"That image is larger than {vault.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)


# -------------------------------------------------------------
# ACCOUNT SETTINGS (every call re-checks the password server-side)
# -------------------------------------------------------------
@app.route("/api/account/verify-password", methods=["POST"])
@login_required
def api_verify_password():
    if Verification.check_password(current_user_id(), body().get("password", "")):
        return jsonify(ok=True)
    return fail("Incorrect password.", 403)


@app.route("/api/account/update", methods=["POST"])
@login_required
def api_account_update():
    data = body()
    ok, message = Verification.update_account(
        current_user_id(), data.get("current_password", ""),
        username=data.get("username"), display_name=data.get("display_name"),
        new_password=data.get("new_password"))
    if not ok:
        return fail(message)
    return jsonify(ok=True, message=message, account=Verification.get_account(current_user_id()))


@app.route("/api/account/email/request", methods=["POST"])
@login_required
def api_email_request():
    data = body()
    if not Verification.check_password(current_user_id(), data.get("current_password", "")):
        return fail("Incorrect password.", 403)
    ok, message = Settings.manage_safety(current_user_id(), data.get("email", ""))
    return jsonify(ok=True, message=message) if ok else fail(message)


@app.route("/api/account/email/verify", methods=["POST"])
@login_required
def api_email_verify():
    ok, message = Settings.confirm_email(current_user_id(), body().get("otp", ""))
    if not ok:
        return fail(message)
    return jsonify(ok=True, message=message, account=Verification.get_account(current_user_id()))


@app.route("/api/account/delete", methods=["POST"])
@login_required
def api_account_delete():
    ok, message = Verification.delete_account(current_user_id(), body().get("current_password", ""))
    if not ok:
        return fail(message, 403)
    session.clear()
    return jsonify(ok=True, message=message)


if __name__ == "__main__":
    classifier.warmup()                      # load the model before serving, not on the first upload
    app.run(debug=True, use_reloader=False, port=8080)

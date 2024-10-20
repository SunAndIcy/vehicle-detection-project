from flask import Blueprint, request, jsonify
from models import db, User  # Assuming the User model is in models.py
from werkzeug.security import generate_password_hash, check_password_hash

register_bp = Blueprint('register', __name__)

# Register user
@register_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not all([username, email, password]):
        return jsonify({"error": "All fields are required"}), 400

    if len(password) < 4:
        return jsonify({"error": "Password must be at least 6 characters long"}), 400

    try:
        # Check if the username or email already exists
        if User.query.filter((User.username == username) | (User.email == email)).first():
            return jsonify({"error": "Username or email already exists"}), 400

        # Use password hashing instead of storing plain text passwords
        password_hash = generate_password_hash(password)

        new_user = User(
            username=username,
            email=email,
            password_hash=password_hash
        )
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"message": "User successfully registered"}), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to register user: {str(e)}"}), 500

# User login
@register_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('usernameOrEmail')
    password = data.get('password')

    if not all([username, password]):
        return jsonify({"error": "All fields are required"}), 400

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password_hash, password):
        return jsonify({"message": "Login successful", "userId": user.id, "username": username}), 200
    else:
        return jsonify({"error": "Invalid username or password"}), 401

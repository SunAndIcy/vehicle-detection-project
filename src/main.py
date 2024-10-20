from flask import Flask
from flask_cors import CORS
from models import db
from app import upload_bp
from auth import register_bp

app = Flask(__name__)
CORS(app)

# Configure the database URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:root@localhost/intelligent_vehicle_system'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Bind SQLAlchemy to the Flask app
db.init_app(app)

# Register blueprints
app.register_blueprint(upload_bp, url_prefix='/api')
app.register_blueprint(register_bp, url_prefix='/auth')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if __name__ == '__main__':
    # Create the upload folder
    import os
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    # Start the Flask app
    app.run(debug=True, host='0.0.0.0')

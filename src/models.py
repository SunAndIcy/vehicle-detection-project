from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# Vehicle Record Table
class ETCVehicleRecord(db.Model):
    __tablename__ = 'etc_vehicle_record'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, nullable=False)  # User ID, foreign key, can be passed during submission or set to a default value
    license_plate = db.Column(db.String(20), nullable=False)
    departure_date = db.Column(db.Date, nullable=False)
    departure_location = db.Column(db.String(100), nullable=False)
    destination_location = db.Column(db.String(100), nullable=True)
    image_url = db.Column(db.String(255), nullable=True)
    status = db.Column(db.SmallInteger, default=1, nullable=False)  # 1: In progress, 2: Trip ended, 3: Pending payment, 4: Completed
    payment_amount = db.Column(db.Numeric(10, 2), nullable=True)  # Payment amount
    gmt_created = db.Column(db.DateTime, default=datetime.utcnow)
    gmt_modified = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# User Table
class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=False)  # Username, must be unique
    email = db.Column(db.String(100), unique=True, nullable=False)  # Email, must be unique
    password_hash = db.Column(db.String(255), nullable=False)  # Password hash
    phone_number = db.Column(db.String(20), nullable=True)  # Contact phone number
    full_name = db.Column(db.String(100), nullable=True)  # User's full name
    address = db.Column(db.String(255), nullable=True)  # User address (optional)
    gmt_created = db.Column(db.DateTime, default=datetime.utcnow)  # Creation time, defaults to current time
    gmt_modified = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # Last modified time

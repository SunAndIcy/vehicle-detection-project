import os
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from utils import load_model, process_image
from models import db, ETCVehicleRecord

upload_bp = Blueprint('upload', __name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Load the YOLOv8 model (cache it in memory to avoid reloading)
model = load_model('/Users/handongdong/pythonProjects/vehicle-detection-project/src/yolov8_model/detect/train4/weights/best.pt')


# Check the file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Process the uploaded file
@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        if filename.endswith(('png', 'jpg', 'jpeg')):
            detections, plate_text = process_image(model, file_path)
        else:
            return jsonify({"error": "Unsupported file format for image"}), 400

        return jsonify({
            "message": "File successfully processed",
            "filename": filename,
            "detections": detections,
            "license_plate": plate_text,
            "image_url": f"{request.host_url}{UPLOAD_FOLDER}/{filename}"
        }), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400


# Submit form information
@upload_bp.route('/submit', methods=['POST'])
def submit_form():
    data = request.form
    departure_date = data.get('departureDate')
    departure_location = data.get('departureLocation')
    license_plate = data.get('licensePlate')
    user_id = data.get('userId')
    image_url = data.get('imageUrl')

    if not all([departure_date, departure_location, license_plate, user_id]):
        return jsonify({"error": "All fields are required"}), 400

    try:
        departure_date_obj = datetime.strptime(departure_date, "%Y-%m-%d").date()

        new_record = ETCVehicleRecord(
            user_id=int(user_id),
            license_plate=license_plate,
            departure_date=departure_date_obj,
            departure_location=departure_location,
            destination_location=None,
            image_url=image_url if image_url else None
        )

        db.session.add(new_record)
        db.session.commit()

        return jsonify({
            "message": "Form data successfully submitted",
            "departureDate": departure_date,
            "departureLocation": departure_location,
            "licensePlate": license_plate,
            "recordId": new_record.id
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to save to database: {str(e)}"}), 500


# Query the list of all ETCVehicleRecord
@upload_bp.route('/records', methods=['GET'])
def get_records():
    try:
        user_id = request.args.get('userId')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        # Query all records
        # records = ETCVehicleRecord.query.all()

        records = ETCVehicleRecord.query.filter(
            ETCVehicleRecord.status == 3,
            ETCVehicleRecord.user_id == user_id
        ).all()

        # Construct the JSON response data
        records_list = []
        for record in records:
            records_list.append({
                "id": record.id,
                "user_id": record.user_id,
                "license_plate": record.license_plate,
                "departure_date": record.departure_date.strftime('%Y-%m-%d'),
                "departure_location": record.departure_location,
                "destination_location": record.destination_location,
                "image_url": record.image_url,
                "status": record.status,
                "payment_amount": str(record.payment_amount),  # Format the payment amount
                "gmt_created": record.gmt_created.strftime('%Y-%m-%d %H:%M:%S'),
                "gmt_modified": record.gmt_modified.strftime('%Y-%m-%d %H:%M:%S') if record.gmt_modified else None
            })

        return jsonify(records_list), 200

    except Exception as e:
        return jsonify({"error": f"Failed to retrieve records: {str(e)}"}), 500

from flask import request

# Query user's unfinished orders
@upload_bp.route('/unfinished-orders', methods=['GET'])
def get_unfinished_orders():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        # Query unfinished orders, assuming status 1 means in progress, 2 means trip ended, 3 means pending payment, 4 means completed
        unfinished_orders = ETCVehicleRecord.query.filter(
            ETCVehicleRecord.user_id == user_id,
            ETCVehicleRecord.status != 3  # Exclude completed orders
        ).all()

        # Construct the JSON response data
        unfinished_orders_list = []
        for order in unfinished_orders:
            unfinished_orders_list.append({
                "id": order.id,
                "license_plate": order.license_plate,
                "departure_date": order.departure_date.strftime('%Y-%m-%d'),
                "departure_location": order.departure_location,
                "destination_location": order.destination_location,
                "status": order.status,
                "payment_amount": str(order.payment_amount) if order.payment_amount else None,
                "gmt_created": order.gmt_created.strftime('%Y-%m-%d %H:%M:%S'),
                "gmt_modified": order.gmt_modified.strftime('%Y-%m-%d %H:%M:%S') if order.gmt_modified else None
            })

        return jsonify(unfinished_orders_list), 200

    except Exception as e:
        return jsonify({"error": f"Failed to retrieve unfinished orders: {str(e)}"}), 500

# End trip endpoint
@upload_bp.route('/end-trip/<int:record_id>', methods=['POST'])
def end_trip(record_id):
    try:
        data = request.get_json()
        destination_location = data.get('destination')

        if not destination_location:
            return jsonify({"error": "Destination location is required"}), 400

        # Retrieve the record by record_id
        record = ETCVehicleRecord.query.get(record_id)

        if not record:
            return jsonify({"error": "Record not found"}), 404

        # Update the destination and status of the record
        record.destination_location = destination_location
        record.status = 2  # Assuming 2 is the pending payment status
        record.payment_amount = 100

        db.session.commit()

        return jsonify({"message": "Trip ended successfully, pending payment"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to end trip: {str(e)}"}), 500

# Payment endpoint
@upload_bp.route('/pay/<int:record_id>', methods=['POST'])
def pay_trip(record_id):
    try:
        # Retrieve the record by record_id
        record = ETCVehicleRecord.query.get(record_id)

        if not record:
            return jsonify({"error": "Record not found"}), 404

        if record.status != 2:  # Check if the trip is pending payment
            return jsonify({"error": "Trip is not pending payment"}), 400

        # Update the record status to 3 (completed)
        record.status = 3
        db.session.commit()

        return jsonify({"message": "Payment successful, trip completed"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to complete payment: {str(e)}"}), 500

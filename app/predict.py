import random

from flask import Blueprint, jsonify, request, current_app


bp = Blueprint("predict", __name__, url_prefix="/predict")


@bp.route("/kp", methods=["POST"])
def predict():
    data = request.json

    return jsonify(
        {
            "image": data["filename"],
            "prediction": [
                {
                    "kp": random.randint(0, 9),
                    "eta": 15,
                }
            ],
        }
    )

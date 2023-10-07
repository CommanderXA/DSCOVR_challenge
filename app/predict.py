import random

import pandas as pd
from flask import Blueprint, jsonify, request, current_app


bp = Blueprint("predict", __name__, url_prefix="/predict")


@bp.route("/kp", methods=["POST"])
def predict():
    data = request.json["data"]

    df = pd.read_csv(data, delimiter=",", header=None)
    print(df.iloc[0, 0])
    return jsonify(
        {
            "timestamp": df.iloc[0, 0],
            "prediction": [
                {
                    "kp": random.randint(0, 9),
                    "eta": 15,
                }
            ],
        }
    )

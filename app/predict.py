from datetime import datetime, timedelta
import random

from flask import Blueprint, jsonify, request, current_app

from nn.inference import forward


bp = Blueprint("predict", __name__, url_prefix="/predict")


@bp.route("/now", methods=["GET"])
def predict_now():
    stream = current_app.config["stream"]
    sample = next(stream)
    data = sample[1]
    print(sample[0])
    kp = forward(current_app.config["model"], data).item()
    return jsonify(
        {
            "timestamp": datetime.fromtimestamp(sample[0].item()),
            "prediction": [
                {
                    "kp": kp,
                    "ground_truth": sample[2].item(),
                    "eta": 15,
                }
            ],
        }
    )


day = 1


@bp.route("/day", methods=["GET"])
def predict_day():
    global day
    stream = current_app.config["stream"]
    for t, x, y in stream:
        dt = datetime.fromtimestamp(t.item())
        if dt > datetime(2023, 1, 1) + timedelta(days=day):
            kp = forward(current_app.config["model"], x).item()
            day += 1

            return jsonify(
                {
                    "timestamp": datetime.fromtimestamp(t.item()),
                    "prediction": [
                        {
                            "kp": kp,
                            "ground_truth": y.item(),
                            "eta": 15,
                        }
                    ],
                }
            )

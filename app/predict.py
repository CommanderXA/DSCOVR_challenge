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
    kp1, kp2 = forward(current_app.config["model"], data)
    return jsonify(
        {
            "timestamp": datetime.fromtimestamp(sample[0].item()),
            "prediction": [
                {
                    "kp": kp1.item(),
                    "ground_truth": sample[2].item(),
                    "eta": 15,
                },
                {
                    "kp": kp2.item(),
                    "ground_truth": sample[3].item(),
                    "eta": 180,
                },
            ],
        }
    )


day = 1


@bp.route("/day", methods=["GET"])
def predict_day():
    global day
    stream = current_app.config["stream"]
    for t, x, y1, y2 in stream:
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
                            "ground_truth": y1.item(),
                            "eta": 15,
                        },
                        {
                            "kp": random.randint(0, 9),
                            "ground_truth": y2.item(),
                            "eta": 180,
                        },
                    ],
                }
            )

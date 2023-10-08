from flask import Flask

from nn.dscovry.model import DSCOVRYModel
from nn.inference import stream_simulation

from . import predict


def create_app(model: DSCOVRYModel, test_config=None):

    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=None
    )
    app.config["model"] = model
    app.config["stream"] = stream_simulation()

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # # ensure the instance folder exists
    # try:
    #     os.makedirs(app.instance_path)
    # except OSError:
    #     pass

    app.register_blueprint(predict.bp)

    return app
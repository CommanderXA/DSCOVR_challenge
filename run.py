from app import create_app

from nn.dscovry.model import DSCOVRYModel

model = DSCOVRYModel()

app = create_app(model)

if __name__ == "__main__":
    app.run(debug=False)

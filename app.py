from flask import Flask, request, render_template
import os
import pickle
import face_recognition
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["STATIC_FOLDER"] = "static"

# Load trained face encodings
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    name = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                matches = face_recognition.compare_faces(data["encodings"], encodings[0])
                if True in matches:
                    name = data["names"][matches.index(True)]
                else:
                    name = "Unknown"
            else:
                name = "No face detected"

    return render_template("index.html", name=name)

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)

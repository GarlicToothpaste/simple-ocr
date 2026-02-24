import os
import json
import glob
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from paddleocr import PPStructureV3

# All paths resolved relative to this file — works regardless of cwd
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for d in (STATIC_DIR, UPLOAD_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)

# Load pipeline once at startup
pipeline = PPStructureV3(
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    enable_mkldnn=False
)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/ocr", methods=["POST"])
def run_ocr():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename) or "upload.png"
    filepath = os.path.join(UPLOAD_DIR, filename)
    file.save(filepath)

    # Clear previous output files so stale results don't bleed through
    for old in glob.glob(os.path.join(OUTPUT_DIR, "*.md")) + \
                glob.glob(os.path.join(OUTPUT_DIR, "*.json")):
        try:
            os.remove(old)
        except OSError:
            pass

    try:
        output = pipeline.predict(filepath)

        for res in output:
            res.save_to_json(save_path=OUTPUT_DIR)
            res.save_to_markdown(save_path=OUTPUT_DIR)

        # Read back generated markdown
        markdown_content = ""
        for md_file in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.md"))):
            with open(md_file, "r", encoding="utf-8") as f:
                markdown_content += f.read() + "\n\n"

        # Read back generated JSON
        json_content = []
        for jf in sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.json"))):
            with open(jf, "r", encoding="utf-8") as f:
                try:
                    json_content.append(json.load(f))
                except Exception:
                    pass

        return jsonify({
            "success": True,
            "markdown": markdown_content.strip(),
            "json": json_content,
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

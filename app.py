import os
import json
import glob
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from paddleocr import PPStructureV3

app = Flask(__name__, static_folder="static")
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

pipeline = PPStructureV3(
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    enable_mkldnn=False
)

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/ocr", methods=["POST"])
def run_ocr():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        output = pipeline.predict(filepath)

        all_results = []
        markdown_content = ""

        for res in output:
            # Save JSON and Markdown
            res.save_to_json(save_path=OUTPUT_FOLDER)
            res.save_to_markdown(save_path=OUTPUT_FOLDER)

            # Collect structured data
            result_data = res.json if hasattr(res, "json") else {}
            all_results.append(result_data)

        # Read generated markdown files
        md_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.md"))
        for md_file in sorted(md_files):
            with open(md_file, "r", encoding="utf-8") as f:
                markdown_content += f.read() + "\n\n"

        # Read generated JSON files
        json_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.json"))
        json_content = []
        for jf in sorted(json_files):
            with open(jf, "r", encoding="utf-8") as f:
                try:
                    json_content.append(json.load(f))
                except Exception:
                    pass

        return jsonify({
            "success": True,
            "markdown": markdown_content.strip(),
            "json": json_content,
            "raw": all_results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
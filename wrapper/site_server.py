from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string, flash
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import cv2
import numpy as np
from detection_functions import detect_image

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.secret_key = 'dev-secret'

INDEX_HTML = '''
<!doctype html>
<title>CV Detection Server</title>
<h1>Upload image for detection</h1>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul>
    {% for m in messages %}
      <li>{{ m }}</li>
    {% endfor %}
    </ul>
  {% endif %}
{% endwith %}
'''

RESULT_HTML = '''
<!doctype html>
<title>Detection result</title>
<h1>Detection result</h1>
<p>Input: <a href="{{ orig_url }}">Original</a></p>
<p>Annotated: <a href="{{ annotated_url }}">Annotated</a></p>
<img src="{{ annotated_url }}" style="max-width:90%;height:auto;">
<br><a href="/">Back</a>
'''


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _draw_boxes_on_image(image_path: str, detections: list, out_path: str):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError('Cannot read image')
    h, w = img.shape[:2]
    for det in detections:
        # support several bbox formats
        x1 = det.get('x1') if det.get('x1') is not None else det.get('xmin') if det.get('xmin') is not None else det.get('left')
        y1 = det.get('y1') if det.get('y1') is not None else det.get('ymin') if det.get('ymin') is not None else det.get('top')
        x2 = det.get('x2') if det.get('x2') is not None else det.get('xmax') if det.get('xmax') is not None else det.get('right')
        y2 = det.get('y2') if det.get('y2') is not None else det.get('ymax') if det.get('ymax') is not None else det.get('bottom')
        bbox = det.get('bbox') or det.get('box')
        if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
        # if any still None, skip
        if None in (x1, y1, x2, y2):
            continue
        try:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        except Exception:
            continue
        # confidence mapping
        conf = det.get('confidence') or det.get('score') or det.get('conf') or det.get('confidence_score')
        try:
            conf = float(conf)
            if conf > 1.0:
                # assume 0-100
                conf = conf / 100.0
        except Exception:
            conf = None
        # color by confidence: low -> red, mid -> yellow, high -> green
        if conf is None:
            color = (0, 255, 0)
            conf_text = ''
        else:
            conf_text = f"{conf*100:.1f}%"
            # BGR: interpolate between red (0,0,255) and green (0,255,0)
            g = int(255 * conf)
            r = int(255 * (1.0 - conf))
            b = 0
            color = (b, g, r)
        label = det.get('class') or det.get('label') or conf_text
        # draw box and filled background for text
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = str(label)
        if conf_text:
            text = f"{text} {conf_text}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx1, ty1 = x1, max(0, y1 - th - 6)
        tx2, ty2 = x1 + tw + 6, ty1 + th + 4
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(img, text, (x1 + 3, ty1 + th + 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(out_path, img)


@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # ensure project data dir exists and copy the uploaded file there so Core can access it
        project_root = Path(__file__).resolve().parents[1]
        data_dir = project_root / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        # create a unique filename to avoid collisions
        import time, shutil
        ts = time.strftime('%Y%m%d_%H%M%S')
        filename_base = Path(filename).stem
        filename_ext = Path(filename).suffix
        data_filename = f"{filename_base}_{ts}{filename_ext}"
        data_path = data_dir / data_filename
        shutil.copyfile(save_path, str(data_path))

        # call detection with a path starting with /data/ so core normalization works
        core_input_path = f"/data/{data_filename}"
        report = detect_image(core_input_path)

        detections = []
        # attempt multiple report formats
        if report is None:
            flash('Detection failed or core not available')
            return redirect(url_for('index'))
        # common format: {'images': [{'objects': [ { 'bbox': [x1,y1,x2,y2], 'class':... }, ... ]}]}
        images = report.get('images') if isinstance(report, dict) else None
        if images and isinstance(images, list) and len(images) > 0:
            objs = images[0].get('objects') or images[0].get('detections')
            if objs:
                for o in objs:
                    # try to map to x1,y1,x2,y2
                    if 'bbox' in o:
                        detections.append({'bbox': o['bbox'], 'class': o.get('class') or o.get('label'), 'confidence': o.get('confidence')})
                    else:
                        detections.append(o)
        # fallback format: top-level 'detections'
        if not detections and isinstance(report.get('detections'), list):
            detections = report.get('detections')

        # Prefer Core's rendered images in output_path (e.g. /results/singles/...), copy if available
        core_output = report.get('output_path') if isinstance(report, dict) else None
        if core_output:
            try:
                # core_output may start with '/results/...'
                out_rel = core_output.lstrip('/')
                core_out_dir = project_root / out_rel
                if core_out_dir.exists() and core_out_dir.is_dir():
                    imgs = list(core_out_dir.glob('**/*.*'))
                    imgs = [p for p in imgs if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
                    # prefer files containing 'annot' or 'out'
                    chosen = None
                    for p in imgs:
                        name = p.name.lower()
                        if 'annot' in name or 'out' in name or 'final' in name:
                            chosen = p
                            break
                    if chosen is None and imgs:
                        chosen = imgs[0]
                    if chosen is not None:
                        # copy chosen to annotated_path for serving
                        import shutil
                        shutil.copyfile(str(chosen), annotated_path)
                        orig_url = url_for('uploaded_file', filename=filename)
                        ann_url = url_for('result_file', filename=os.path.basename(annotated_path))
                        return render_template_string(RESULT_HTML, orig_url=orig_url, annotated_url=ann_url)
            except Exception:
                pass

        annotated_name = f"annotated_{data_filename}"
        annotated_path = os.path.join(app.config['RESULTS_FOLDER'], annotated_name)
        try:
            _draw_boxes_on_image(str(data_path), detections, annotated_path)
        except Exception:
            # if drawing failed, copy original to annotated path
            import shutil
            shutil.copyfile(str(data_path), annotated_path)

        orig_url = url_for('uploaded_file', filename=filename)
        ann_url = url_for('result_file', filename=annotated_name)
        return render_template_string(RESULT_HTML, orig_url=orig_url, annotated_url=ann_url)
    else:
        flash('Invalid file type')
        return redirect(url_for('index'))


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<path:filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

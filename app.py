from PIL import Image
import base64
import io
from flask import Flask, get_flashed_messages, redirect, render_template, request, flash, session
from palette import create_palette

app = Flask(__name__, template_folder='templates')
app.secret_key = 'cool-secret-key'

@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


images_folder = 'images/'


@app.route("/palette", methods=["GET", "POST"])
def palette():
    if request.method == 'POST':
        f = request.files['file']
        if(f.filename == ''):
            flash('No file selected', 'error')
            return redirect('/')
        encoded = base64.b64encode(f.read()).decode('utf-8')
        palette, colors, width, height = create_palette(f)
        palette_data = io.BytesIO()
        palette.save(palette_data, format='JPEG')
        palette_encoded = base64.b64encode(palette.getvalue()).decode('utf-8')
        info = {
            'image': encoded,
            'palette': palette_encoded,
            'colors': colors,
            'filename': f.filename,
            'width': width,
            'height': height
        }
        return render_template("palette.html", info=info)
    else:
        return render_template("404.html"), 404

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404



if __name__ == "__main__":
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.run(debug=True)

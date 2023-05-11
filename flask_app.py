from flask import Flask, render_template, send_from_directory, url_for, request
from ultralytics import YOLO
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
import os
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfasfdgdf'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

model = YOLO('yolo_custom.pt')

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Только изображения'),
            FileRequired('Поле файла не должно быть пустым')
        ]
    )
    submit = SubmitField('Распознать')


@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        start_time = time.perf_counter()
        model.predict(source=f"uploads/{filename}", show=True, save=True, conf=0.5, line_thickness=1, hide_labels=True,
                      hide_conf=True)
        end_time = (time.perf_counter() - start_time) * 1000
        elapsed_time_ms = f"{end_time:.1f}ms"
        os.remove(f"uploads/{filename}")
        os.rename(f"runs/detect/predict/{filename}", f"uploads/{filename}")
        os.removedirs("runs/detect/predict")
    else:
        elapsed_time_ms = None
        file_url = None
    return render_template('index.html', form=form, file_url=file_url, elapsed_time_ms=elapsed_time_ms)


if __name__ == '__main__':
    app.run(debug=True)
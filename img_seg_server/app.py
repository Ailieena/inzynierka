from pathlib import Path
import shutil
from PIL import Image
import cv2
import keras
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, flash, redirect

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded/'
app.secret_key = "secretkey"

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER_DIR = Path(app.config['UPLOAD_FOLDER'])
USER_IMAGE_PATH = UPLOAD_FOLDER_DIR / 'uploaded-img.jpg'
CONVERTED_IMAGE_PATH = UPLOAD_FOLDER_DIR / 'converted-img.png'
UPLOAD_FOLDER_DIR.mkdir(exist_ok=True, parents=True)
model = keras.models.load_model('../model3.keras')

def get_ext(filename: str):
    return filename.rsplit('.', 1)[1].lower()


def allowed_file(filename: str):
    return '.' in filename and \
           get_ext(filename) in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    kwargs = {}
    if USER_IMAGE_PATH.exists():
        kwargs['user_image'] = str(USER_IMAGE_PATH)

    if CONVERTED_IMAGE_PATH.exists():
        kwargs['converted_image'] = str(CONVERTED_IMAGE_PATH)

    return render_template('index.html', **kwargs)


def post_uplaod_image():
    if 'file' not in request.files:
        flash('No file part', 'warning')
        return redirect('/')

    file = request.files['file']
    if file.filename == '':
        flash('Missing file', 'warning')
        return redirect('/')

    if not allowed_file(file.filename):
        exts = ', '.join(ALLOWED_EXTENSIONS)
        flash(f'Invalid file extension, allowed: {exts}', 'warning')
        return redirect('/')

    file.save(USER_IMAGE_PATH)
    flash('Image was successfully uploaded', 'success')
    return redirect('/')


def get_uploaded_image():
    if not USER_IMAGE_PATH.exists():
        flash('Firstly you need to upload image', 'warning')
    return redirect('/')


@app.route('/user-image', methods=['GET', 'POST'])
def user_image():
    if request.method == 'POST':
        return post_uplaod_image()
    else:
        return get_uploaded_image()


def delete_background(image, mask, file_res):
    mask = tf.image.resize(mask, file_res)
    mask = (tf.squeeze(mask, axis=-1))
    mask = tf.cast(mask, tf.dtypes.uint8).numpy()
    image = image.numpy()
    result = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    result[:, :, 3] = mask
    return result

def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def normalize_image(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def parse_image_for_pred(image):
    IMG_SIZE = 512, 512
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.expand_dims(image, axis = 0)
    return image

def to_png(image_arr, CONVERTED_IMAGE_PATH):
    image = Image.fromarray((image_arr * 255).astype(np.uint8))
    image.save(CONVERTED_IMAGE_PATH)

def remove_background(image_path):
    image_content = tf.io.read_file(str(image_path))
    image = tf.image.decode_jpeg(image_content, channels=3)

    prepared_image = parse_image_for_pred(image)
    pred_mask = model.predict(prepared_image)
    pred_mask = create_mask(pred_mask)

    file_res = image.shape[:-1]
    image = normalize_image(image)
    new_image = delete_background(image, pred_mask, file_res)
    # new_image = normalize_image(new_image)
    to_png(new_image, CONVERTED_IMAGE_PATH)

@app.route('/remove-bg', methods=['POST'])
def remove_bg():
    if not USER_IMAGE_PATH.exists():
        flash('You need to upload image first', 'warning')
        return redirect('/')

    remove_background(USER_IMAGE_PATH,)
    return redirect('/')


@app.route('/_clear', methods=['GET'])
def clear():
    USER_IMAGE_PATH.unlink(missing_ok=True)
    CONVERTED_IMAGE_PATH.unlink(missing_ok=True)
    return redirect('/')

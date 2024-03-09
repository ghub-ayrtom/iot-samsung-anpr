import sys
sys.path.insert(1, '/home/ayrtom/PycharmProjects/iot-samsung-anpr/src/models')

from flask_bcrypt import Bcrypt
from src.models.Decoders import BeamDecoder, decode_function
from flask_login import LoginManager, login_required, login_user, logout_user, UserMixin
import cv2
from flask import Flask, redirect, render_template, request, url_for
from flask_wtf import FlaskForm
from sqlalchemy import ForeignKey, PickleType
from PIL import Image
from wtforms.validators import InputRequired, Length, ValidationError
import io
from typing import List
from src.models.LPRNet import load_default_lprnet
from src.models.SpatialTransformer import load_default_stn
import logging
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.ext.mutable import MutableList
import numpy as np
import os
from wtforms import PasswordField, StringField, SubmitField
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError
import torch
from ultralytics import YOLO


database_path = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(database_path, 'local_database/sqlite.db')
app.config['SECRET_KEY'] = 'thisisasecretkey'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

sqlite_database = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

upload_images_path = 'uploaded_images'
port = 8080

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

license_plate_detection_model = YOLO('../../models/LPDNet.pt')
license_plate_recognition_model = load_default_lprnet(device)
spatial_transformer_model = load_default_stn(device)


@login_manager.user_loader
def load_client(client_id):
    return Client.query.get(int(client_id))


class Client(sqlite_database.Model, UserMixin):
    __tablename__ = 'clients'

    id = sqlite_database.Column(sqlite_database.Integer, primary_key=True)
    username = sqlite_database.Column(sqlite_database.String(15), nullable=False, unique=True)
    password = sqlite_database.Column(sqlite_database.String(25), nullable=False)
    company_name = sqlite_database.Column(sqlite_database.String(25), nullable=False, unique=True)
    barriers: Mapped[List['Barrier']] = relationship()

    def __repr__(self):
        return f'<Client: {self.name}>'


class Barrier(sqlite_database.Model):
    __tablename__ = 'barriers'

    id = sqlite_database.Column(sqlite_database.Integer, primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey('clients.id'), nullable=False)
    model = sqlite_database.Column(sqlite_database.String(15))
    client_company_name = sqlite_database.Column(sqlite_database.String(25), nullable=False)
    location = sqlite_database.Column(sqlite_database.String(50), nullable=False)
    license_plates = sqlite_database.Column(MutableList.as_mutable(PickleType), default=[])

    def __repr__(self):
        return f'<{self.client_company_name}\'s barrier: {self.model} ({self.location})>'


class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=3, max=15)], render_kw={'placeholder': 'Username'})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=25)], render_kw={'placeholder': 'Password'})
    company_name = StringField(validators=[InputRequired(), Length(min=3, max=25)],
                               render_kw={'placeholder': 'Company Name'})
    submit = SubmitField('Sign Up')

    def validate_input(self, username, company_name):
        username_already_exists = Client.query.filter_by(username=username.data).first()
        company_name_already_exists = Client.query.filter_by(title=company_name.data).first()

        if username_already_exists:
            raise ValidationError('That username already exists. Please choose a different one.')
        elif company_name_already_exists:
            raise ValidationError('That company name already exists. Please choose a different one.')


class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=3, max=15)], render_kw={'placeholder': 'Username'})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=25)], render_kw={'placeholder': 'Password'})
    submit = SubmitField('Sign In')


class TagListField(StringField):
    """Stringfield for a list of separated tags"""

    def __init__(self, label='', validators=None, remove_duplicates=True, separator=' ', **kwargs):
        """
        Construct a new field.
        :param label: The label of the field.
        :param validators: A sequence of validators to call when validate is called.
        :param remove_duplicates: Remove duplicates in a case-insensitive manner.
        :param to_lowercase: Cast all values to lowercase.
        :param separator: The separator that splits the individual tags.
        """
        super(TagListField, self).__init__(label, validators, **kwargs)
        self.remove_duplicates = remove_duplicates
        self.separator = separator
        self.data = []

    def _value(self):
        if self.data:
            return u', '.join(self.data)
        else:
            return u''

    def process_formdata(self, valuelist):
        if valuelist:
            self.data = [x.strip() for x in valuelist[0].split(self.separator)]
            if self.remove_duplicates:
                self.data = list(self._remove_duplicates(self.data))

    @classmethod
    def _remove_duplicates(cls, seq):
        """Remove duplicates in a case-insensitive, but case preserving manner"""
        d = {}
        for item in seq:
            if item.lower() not in d:
                d[item.lower()] = True
                yield item


class AddBarrierForm(FlaskForm):
    model = StringField(validators=[Length(min=3, max=15)], render_kw={'placeholder': 'Model'})
    location = StringField(validators=[InputRequired(), Length(min=5, max=50)], render_kw={'placeholder': 'Location'})
    license_plates = TagListField('License Plates', separator=',')
    submit = SubmitField('Add')


def get_images_count(path):
    images_count = 1

    for image_file_name in os.listdir(path):
        if os.path.isfile(os.path.join(path, image_file_name)):
            images_count += 1

    return images_count


def convert_image_bytes_to_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def get_prediction(image_bytes, debug=False):
    license_plate_image = convert_image_bytes_to_image(image_bytes)

    """
        conf: устанавливает минимальный порог уверенности для обнаружения (false positives reduce, 0.25 - default)
        iou: более высокие значения приводят к уменьшению количества обнаружений за счёт устранения перекрывающихся 
        боксов, что полезно для уменьшения количества дубликатов (multiply detections reduce, 0.7 - default)
        max_det: максимальное количество обнаружений, допустимое для одного изображения (300 - default)
    """
    license_plate_detections = license_plate_detection_model.predict(license_plate_image, conf=0.25, iou=0.7, max_det=300)

    for license_plate in license_plate_detections:
        for license_plate_bounding_box in license_plate.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = license_plate_bounding_box

            license_plate_image_cropped = license_plate_image[int(y1):int(y2), int(x1):int(x2)]
            license_plate_image_cropped = cv2.resize(license_plate_image_cropped, (94, 24),
                                                     interpolation=cv2.INTER_CUBIC)
            license_plate_image_cropped = (np.transpose(np.float32(license_plate_image_cropped),
                                                        (2, 0, 1)) - 127.5) * 0.0078125

            data = torch.from_numpy(license_plate_image_cropped).float().unsqueeze(0).to(device)

            license_plate_image_transformed = spatial_transformer_model(data)
            predictions = license_plate_recognition_model(license_plate_image_transformed)
            predictions = predictions.cpu().detach().numpy()

            labels, probability, predicted_labels = decode_function(
                predictions,
                [
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T',
                    'Y', 'X', '-',
                ],
                BeamDecoder
            )

            if (probability[0] < -85) and (len(labels[0]) in [8, 9]):
                barriers_table = Barrier.query.all()

                # todo: Поиск преграждения с которого пришло изображение, например, по его месторасположению

                for barrier in barriers_table:
                    for barrier_license_plate in barrier.license_plates:
                        # Если распознанный автомобильный номер найден в соответствующей таблице локальной базы данных
                        if labels[0] == barrier_license_plate:
                            if debug:
                                uploaded_images_count = get_images_count(upload_images_path)
                                save_location = (
                                    os.path.join(
                                        app.root_path, '{}/{}.jpg'.format(upload_images_path, uploaded_images_count)
                                    )
                                )

                                cv2.rectangle(
                                    license_plate_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5
                                )
                                cv2.putText(
                                    license_plate_image, labels[0], (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3
                                )
                                cv2.imwrite(save_location, license_plate_image)

                            return 'OPEN'  # Подаём команду на открытие преграждения

    return 'CLOSE'  # Иначе, подаём команду на закрытие преграждения


@app.route('/recognize', methods=['POST'])
def recognize_license_plate(debug=False):
    if request.method == 'POST':
        license_plate_image_raw_bytes = request.get_data()

        if debug:
            uploaded_images_count = get_images_count(upload_images_path)
            save_location = (os.path.join(app.root_path, '{}/{}.jpg'.format(upload_images_path, uploaded_images_count)))

            license_plate_image_file = open(save_location, 'wb')
            license_plate_image_file.write(license_plate_image_raw_bytes)
            license_plate_image_file.close()

        license_plate_text = get_prediction(license_plate_image_raw_bytes)
        return license_plate_text


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        client = Client.query.filter_by(username=form.username.data).first()

        if client:
            if bcrypt.check_password_hash(client.password, form.password.data):
                login_user(client)
                return redirect(url_for('dashboard', client_id=client.id, company=client.company_name))

    return render_template('login.html', form=form)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/barriers', methods=['GET', 'POST'])
@login_required
def barriers():
    client_id = request.args.get('client_id')
    client_barriers_list = Barrier.query.filter_by(client_id=client_id).all()
    return render_template('barriers.html', barriers=client_barriers_list)


@app.route('/barriers/add', methods=['GET', 'POST'])
@login_required
def add_barrier():
    form = AddBarrierForm()
    client_id = request.args.get('client_id')

    if form.validate_on_submit():
        new_barrier = Barrier(
            client_id=client_id,
            model=form.model.data,
            client_company_name=request.args.get('company'),
            location=form.location.data,
            license_plates=form.license_plates.data,
        )

        try:
            sqlite_database.session.add(new_barrier)
            sqlite_database.session.commit()
            return redirect(url_for('barriers', client_id=client_id))
        except SQLAlchemyError as error:
            sqlite_database.rollback()
            logging.error('Failed to commit changes because of {error}. Doing rollback...'.format(error=error))

    return render_template('add_barrier.html', form=form)


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        new_client = Client(
            username=form.username.data,
            password= bcrypt.generate_password_hash(form.password.data),
            company_name=form.company_name.data,
        )

        try:
            sqlite_database.session.add(new_client)
            sqlite_database.session.commit()
            return redirect(url_for('login'))
        except SQLAlchemyError as error:
            sqlite_database.rollback()
            logging.error('Failed to commit changes because of {error}. Doing rollback...'.format(error=error))

    return render_template('register.html', form=form)


if __name__ == '__main__':
    with app.app_context():
        sqlite_database.create_all()

    app.run(host='0.0.0.0', port=port, debug=True)

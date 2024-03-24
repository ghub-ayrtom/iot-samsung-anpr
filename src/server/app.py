import sys
sys.path.insert(1, '/home/ayrtom/PycharmProjects/iot-samsung-anpr/src/models')

from transliterate.discover import autodiscover
import base64
from flask_bcrypt import Bcrypt
from src.models.Decoders import BeamDecoder, decode_function
import cv2
from datetime import datetime
from flask import Flask, redirect, render_template, request, url_for
from flask_wtf import FlaskForm
from PIL import Image
from wtforms.validators import InputRequired, Length
import io
from src.models.LPRNet import load_default_lprnet
from src.models.SpatialTransformer import load_default_stn
import logging
from flask_login import LoginManager, login_required, login_user, logout_user, UserMixin
import math
from sqlalchemy.ext.mutable import MutableList
import numpy as np
import os
from wtforms import PasswordField, StringField, SubmitField
from sqlalchemy import PickleType
from transliterate.base import registry, TranslitLanguagePack
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError
import torch
from transliterate import translit
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
    barriers = sqlite_database.Column(MutableList.as_mutable(PickleType), default=[])


class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=3, max=15)])
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=25)])
    company_name = StringField(validators=[InputRequired(), Length(min=3, max=25)])
    submit = SubmitField('Зарегистрироваться')

    def validate_input(self, username, company_name):
        username_already_exists = Client.query.filter_by(username=username.data).first()
        company_name_already_exists = Client.query.filter_by(company_name=company_name.data).first()

        if username_already_exists:
            return 1
        elif company_name_already_exists:
            return -1
        else:
            return 0


class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=3, max=15)])
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=25)])
    submit = SubmitField('Авторизоваться')


class LicensePlatesLanguagePack(TranslitLanguagePack):
    language_code = 'lp'
    language_name = "License Plates"
    mapping = (
        u'АВЕКМНОРСТУХ',  # Кириллица
        u'ABEKMHOPCTYX',  # Латиница
    )


class TagListField(StringField):
    def __init__(self, min_length=1, max_length=math.inf, remove_duplicates=True, to_uppercase=True,
                 transliterate=False, separator=',', **kwargs):
        super(TagListField, self).__init__(**kwargs)
        self.min_length = min_length,
        self.max_length = max_length,
        self.remove_duplicates = remove_duplicates
        self.to_uppercase = to_uppercase
        self.transliterate = transliterate
        self.separator = separator
        self.data = []

    def _value(self):
        if self.data:
            return u', '.join(self.data)
        else:
            return u''

    def process_formdata(self, tags):
        if tags:
            # Делим по запятым целую введённую строку на отдельные теги и удаляем в них лишние пробелы,
            # если они имеют заданную длину
            self.data = [
                tag.strip() if self.min_length[0] <= len(tag.strip()) <= self.max_length[0] else None
                for tag in tags[0].split(self.separator)
            ]

            if self.remove_duplicates:
                # Удаляем дублирующиеся теги
                self.data = list(self._remove_duplicates(self.data, self.transliterate))

            if self.to_uppercase:
                # Переводим все теги в верхний регистр
                self.data = [tag.upper() for tag in self.data]

            # Если в качестве тегов в поле вводятся автомобильные номера
            if self.transliterate:
                self.data = sorted(self.data)  # Сортируем их в алфавитном порядке

    @classmethod
    def _remove_duplicates(cls, tags, transliterate):
        duplicate_tags = {}

        for tag in tags:
            if tag is not None:
                if transliterate:
                    autodiscover()  # Даёт возможность использовать встроенные языковые пакеты со своими собственными
                    # Принудительно регистрируем свой собственный языковой пакет
                    registry.register(LicensePlatesLanguagePack, force=True)
                    # Преобразуем теги автомобильных номеров в их латинское написание для корректного сравнения
                    tag = translit(tag.upper(), 'lp')

                if tag not in duplicate_tags:
                    duplicate_tags[tag] = True
                    yield tag


class AddBarrierForm(FlaskForm):
    model = StringField()
    location = StringField(validators=[Length(max=50)])
    license_plates = TagListField(min_length=8, max_length=9, transliterate=True, validators=[InputRequired()],
                                  render_kw={'placeholder': 'B776YC77, E015HA73, ...'})
    events = TagListField(to_uppercase=False, render_kw={'placeholder': 'Открытие, Закрытие, ...'})
    submit = SubmitField('Добавить')


class EditBarrierForm(FlaskForm):
    model = StringField()
    location = StringField(validators=[Length(max=50)])
    license_plates = TagListField(min_length=8, max_length=9, transliterate=True, validators=[InputRequired()],
                                  render_kw={'placeholder': 'B776YC77, E015HA73, ...'})
    events = TagListField(to_uppercase=False, render_kw={'placeholder': 'Открытие, Закрытие, ...'})
    submit = SubmitField('Внести изменения')


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


def convert_numpy_array_to_image(numpy_array):
    _, image_buffer = cv2.imencode('.png', numpy_array)
    return base64.b64encode(image_buffer).decode('utf-8')


def get_prediction(client_company_name, client_barrier_id, image_bytes):
    client_barriers = Client.query.filter_by(company_name=client_company_name).first().barriers
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
                for barrier_license_plate in client_barriers[client_barrier_id - 1]['license_plates']:
                    # Если распознанный автомобильный номер найден в соответствующем списке для данного преграждения
                    if labels[0] == barrier_license_plate:
                        cv2.rectangle(
                            license_plate_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5
                        )
                        cv2.putText(
                            license_plate_image, labels[0], (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3
                        )

                        barrier_log = {
                            'record_number': len(client_barriers[client_barrier_id - 1]['logs']) + 1 if len(
                                client_barriers[client_barrier_id - 1]['logs']) != 0 else 1,
                            'event': '',
                            'datetime': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                            'frame': convert_numpy_array_to_image(license_plate_image),
                        }

                        client_barriers[client_barrier_id - 1]['logs'].append(barrier_log)

                        try:
                            Client.query.filter_by(company_name=client_company_name).update(
                                {'barriers': client_barriers})
                            sqlite_database.session.commit()
                            return 'OPEN'  # Подаём команду на открытие преграждения
                        except SQLAlchemyError as error:
                            sqlite_database.rollback()
                            logging.error(
                                'Failed to commit changes because of {error}. Doing rollback...'.format(error=error))

    barrier_log = {
        'record_number': len(client_barriers[client_barrier_id - 1]['logs']) + 1 if len(
            client_barriers[client_barrier_id - 1]['logs']) != 0 else 1,
        'event': '',
        'datetime': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        'frame': convert_numpy_array_to_image(license_plate_image),
    }

    client_barriers[client_barrier_id - 1]['logs'].append(barrier_log)

    try:
        Client.query.filter_by(company_name=client_company_name).update({'barriers': client_barriers})
        sqlite_database.session.commit()
        return 'CLOSE'  # Иначе, подаём команду на закрытие преграждения
    except SQLAlchemyError as error:
        sqlite_database.rollback()
        logging.error('Failed to commit changes because of {error}. Doing rollback...'.format(error=error))


@app.route('/recognize', methods=['POST'])
def recognize_license_plate(debug=False):
    if request.method == 'POST':
        client_company_name = request.headers['Company-Name']
        client_barrier_id = int(request.headers['Barrier-ID'])
        license_plate_image_raw_bytes = request.get_data()

        if debug:
            uploaded_images_count = get_images_count(upload_images_path)
            save_location = (os.path.join(app.root_path, '{}/{}.jpg'.format(upload_images_path, uploaded_images_count)))

            license_plate_image_file = open(save_location, 'wb')
            license_plate_image_file.write(license_plate_image_raw_bytes)
            license_plate_image_file.close()

        barrier_action = get_prediction(client_company_name, client_barrier_id, license_plate_image_raw_bytes)
        return barrier_action


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    validation_error_message = ''

    if form.validate_on_submit():
        client = Client.query.filter_by(username=form.username.data).first()

        if client:
            if bcrypt.check_password_hash(client.password, form.password.data):
                login_user(client)
                return redirect(url_for('dashboard', client_id=client.id, client_company_name=client.company_name))
            else:
                validation_error_message = 'Введён неверный пароль!'
        else:
            validation_error_message = 'Пользователь с таким именем не зарегистрирован!'

    return render_template('login.html', form=form, error=validation_error_message)


@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/barriers', methods=['GET', 'POST'])
@login_required
def barriers():
    client = Client.query.get(request.args.get('client_id'))
    return render_template('barriers.html', barriers=client.barriers)


@app.route('/barriers/add', methods=['GET', 'POST'])
@login_required
def add_barrier():
    form = AddBarrierForm()
    client = Client.query.get(request.args.get('client_id'))

    if form.validate_on_submit():
        new_barrier = {
            'id': client.barriers[-1]['id'] + 1 if len(client.barriers) > 0 else 1,
            'model': form.model.data,
            'location': form.location.data,
            'license_plates': form.license_plates.data,
            'events': form.events.data,
            'logs': [],
        }

        client.barriers.append(new_barrier)

        try:
            sqlite_database.session.query(Client).filter(Client.id == client.id).update({'barriers': client.barriers})
            sqlite_database.session.commit()
            return redirect(url_for('barriers', client_id=client.id))
        except SQLAlchemyError as error:
            sqlite_database.rollback()
            logging.error('Failed to commit changes because of {error}. Doing rollback...'.format(error=error))

    return render_template('add_barrier.html', form=form)


@app.route('/barriers/delete', methods=['POST'])
@login_required
def delete_barrier():
    client_barriers = Client.query.get_or_404(request.args.get('client_id')).barriers

    for barrier in client_barriers:
        if barrier['id'] == int(request.args.get('barrier_id')):
            client_barriers.remove(barrier)

    try:
        sqlite_database.session.query(Client).filter(Client.id == request.args.get('client_id')).update({
            'barriers': client_barriers
        })
        sqlite_database.session.commit()
        return redirect(url_for('barriers', client_id=request.args.get('client_id')))
    except SQLAlchemyError as error:
        sqlite_database.rollback()
        logging.error('Failed to commit changes because of {error}. Doing rollback...'.format(error=error))


@app.route('/barriers/edit', methods=['GET', 'POST'])
@login_required
def edit_barrier():
    form = EditBarrierForm()
    client_barriers = Client.query.get_or_404(request.args.get('client_id')).barriers

    editable_barrier = {}
    editable_barrier_index = -1

    for index, barrier in enumerate(client_barriers):
        if barrier['id'] == int(request.args.get('barrier_id')):
            editable_barrier = barrier
            editable_barrier_index = index

    if request.method == 'POST':
        if form.validate_on_submit():
            editable_barrier['model'] = form.model.data
            editable_barrier['location'] = form.location.data
            editable_barrier['license_plates'] = form.license_plates.data
            editable_barrier['events'] = form.events.data

            client_barriers[editable_barrier_index] = editable_barrier

            try:
                sqlite_database.session.query(Client).filter(Client.id == request.args.get('client_id')).update({
                    'barriers': client_barriers
                })
                sqlite_database.session.commit()
                return redirect(url_for('barriers', client_id=request.args.get('client_id')))
            except SQLAlchemyError as error:
                sqlite_database.rollback()
                logging.error('Failed to commit changes because of {error}. Doing rollback...'.format(error=error))
    else:
        form.model.data = editable_barrier['model']
        form.location.data = editable_barrier['location']
        form.license_plates.data = editable_barrier['license_plates']
        form.events.data = editable_barrier['events']

        return render_template('edit_barrier.html', form=form, barrier=editable_barrier)


@app.route('/barriers/logs', methods=['GET', 'POST'])
@login_required
def barrier_logs():
    client_barriers = Client.query.get_or_404(request.args.get('client_id')).barriers

    for barrier in client_barriers:
        if barrier['id'] == int(request.args.get('barrier_id')):
            return render_template('barrier_logs.html', barrier=barrier)


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    validation_error_message = ''

    if form.validate_on_submit():
        if form.validate_input(form.username, form.company_name) == 0:
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
        elif form.validate_input(form.username, form.company_name) == 1:
            validation_error_message = 'Пользователь с таким именем уже зарегистрирован!'
        elif form.validate_input(form.username, form.company_name) == -1:
            validation_error_message = 'Компания с таким названием уже зарегистрирована!'

    return render_template('register.html', form=form, error=validation_error_message)


if __name__ == '__main__':
    with app.app_context():
        sqlite_database.create_all()

    app.run(host='0.0.0.0', port=port, debug=True)

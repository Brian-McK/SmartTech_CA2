# this file will have websockets that allow us to send image from simulator to model, the model will then return the steering angle and throttle and then we can use autonomous mode to control the car in the simulator

import socketio
import eventlet
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from matplotlib import image as mpimg

sio = socketio.Server()
app = Flask(__name__)

speed_limit = 20


def preprocess_img(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255
    return img


# sends json to the simulator
def send_control(steering_angle, throttle):
    sio.emit('steer', data={'steering_angle': steering_angle.__str__(), 'throttle': throttle.__str__()})


# get from simulator
@sio.on('telemetry')
def telemetry(sid, data):
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    image = preprocess_img(image)
    image = np.array([image])
    speed = float(data['speed'])
    throttle = 1.0 - speed / speed_limit
    steering_angle = float(model.predict(image))
    send_control(steering_angle, throttle)


# happens when connected to the simulator
@sio.on('connect')
def connect(sid, environ):
    print("Connected")
    send_control(0, 0)


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

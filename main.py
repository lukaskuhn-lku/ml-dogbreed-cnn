import tensorflow as tf
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow import keras
from breeds import breeds
import falcon
import json

#Const
IMG_SIZE = 160

#Load Model
model = keras.models.load_model("./dog_breed.h5")

def get_dog_breed_by_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image = tf.cast(np.asarray(image), tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    predictions = model.predict(np.array( [image,] ))

    predicted_label = breeds[np.argmax(predictions[0])]
    predicted_percent = predictions[0][np.argmax(predictions[0])]

    return predicted_label, predicted_percent

class BreedResource(object):
    def on_get(self, req, resp):
        url = None
        img_as_array = None
        label = None
        percent = None

        for key, value in req.params.items():
                if key == "url":
                    url = value

        if url != None:
            label, percent = get_dog_breed_by_url(url)

        doc = {
            'breed': [
                {
                    'name': str(label),
                    'percent': str(percent)
                }
            ]
        }

        resp.body = json.dumps(doc, ensure_ascii=False)
        resp.status = falcon.HTTP_200

app = falcon.API()
breed = BreedResource()
app.add_route('/breed', breed)

#label, percent = get_dog_breed_by_url("https://www.deine-tierwelt.de/magazin/wp-content/uploads/sites/2/2018/07/Border-Collie.jpg")
#print(label)
#print(percent)


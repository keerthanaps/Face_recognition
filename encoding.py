import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
keerthana_image = face_recognition.load_image_file("keerthana.jpg")
keerthana_face_encoding = face_recognition.face_encodings(keerthana_image)[0]

# Load a second sample picture and learn how to recognize it.
drishya_image = face_recognition.load_image_file("drishya.jpg")
drishya_face_encoding = face_recognition.face_encodings(drishya_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    keerthana_face_encoding,
    dridhya_face_encoding
]
known_face_names = [
    "Keerthana",
    "Drishya"
]
print('Learned encoding for', len(known_face_encodings), 'images.')

import tensorflow as tf
import numpy as np
from keras.preprocessing import image  # type: ignore

def prepare(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)  

array = ["Array with your classes"]

interpreter = tf.lite.Interpreter(model_path="<Path to your model>")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

image_path = "Path to your image"
try:
    input_data = prepare(image_path)
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit(1)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
# printing 
print(array[np.argmax(output_data,axis=1)[0]])

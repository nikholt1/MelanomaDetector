from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from io import BytesIO
from threading import Thread
from queue import Queue

app = Flask(__name__)

# Load Keras model once
model = tf.keras.models.load_model("./model/melanoma_model.keras")

# Queue for transferring images from Flask to Tkinter
image_queue = Queue()


@app.route('/cnn/return', methods=['POST'])
def return_verdict():

    image_bytes = request.get_data()

    if not image_bytes:
        return jsonify({'error': 'No image received'}), 400

    print(f"Received image: {len(image_bytes)} bytes")

    # Send image to Tkinter
    image_queue.put(image_bytes)

    # Convert image bytes to PIL image
    image = Image.open(BytesIO(image_bytes))

    # Preprocess image for CNN
    image = image.convert("RGB").resize((224, 224))
    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Make CNN prediction
    prediction = float(model.predict(image, verbose=0)[0][0])

    print(f"Malignant score: {prediction:.4f}")

    # Return raw prediction value
    return jsonify(prediction)


# Tkinter window
def start_gui():

    root = tk.Tk()
    root.title("CNN Image Scanner")
    root.geometry("330x450")

    image_label = tk.Label(root)
    image_label.pack(expand=True)

    def update_image():

        if not image_queue.empty():

            image_bytes = image_queue.get()

            pil_image = Image.open(BytesIO(image_bytes))
            pil_image.thumbnail((550, 450))

            photo = ImageTk.PhotoImage(pil_image)

            image_label.config(image=photo)
            image_label.image = photo

            print("Image displayed!")

        root.after(100, update_image)

    update_image()
    root.mainloop()


if __name__ == '__main__':

    server = Thread(
        target=lambda: app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False
        ),
        daemon=True
    )

    server.start()

    start_gui()
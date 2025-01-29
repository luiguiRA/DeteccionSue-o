import cv2
import numpy as np
from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
import vlc
import threading

app = Flask(__name__)

# Cargar el modelo de predicción CAMBIO DE RUTA PARA EJECUTAR EN LOCAL WILMER
model = load_model(r"C:\Users\gueva\DeteccionSue-o-main\DeteccionSue-o-main\keras_model1.h5")


# Cargar el clasificador de ojos
eyeLeft = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
eyeRight = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# Configuración del video
video_capture = cv2.VideoCapture(0)
p = vlc.MediaPlayer("wakeup.mp3")

# Variables de contador
contador = 0
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Configurar la resolución de la cámara
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Ancho de 640 píxeles
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Alto de 480 píxeles

def predict_sleep(frame, height, width):
    """Realiza la predicción de sueño solo cuando sea necesario."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Identificar ojos
    ojo_der = eyeRight.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3, minSize=(30, 30))
    ojo_izq = eyeLeft.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3, minSize=(30, 30))

    left_x, left_y, left_w, left_h = 0, 0, 0, 0
    right_x, right_y, right_w, right_h = 0, 0, 0, 0

    for (x, y, w, h) in ojo_der:
        right_x, right_y, right_w, right_h = x, y, w, h
        break
    for (x, y, w, h) in ojo_izq:
        left_x, left_y, left_w, left_h = x, y, w, h
        break

    if (left_x > right_x):
        start_x, end_x = right_x, (left_x + left_w)
    else:
        start_x, end_x = left_x, (right_x + right_w)

    if (left_y > right_y):
        start_y, end_y = right_y, (left_y + left_h)
    else:
        start_y, end_y = left_y, (right_y + right_h)

    # Detección de sueño
    if ((end_x - start_x) > 120 and (end_y - start_y) < 200):
        start_x, start_y, end_x, end_y = start_x - 30, start_y - 50, end_x + 30, end_y + 50
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        img = frame[start_y:end_y, start_x:end_x]
        imagen = cv2.resize(img, (224, 224))
        imagen_normalizada = (imagen.astype(np.float32) / 127.0) - 1
        data[0] = imagen_normalizada
        prediction = model.predict(data)

        if (list(prediction)[0][1] >= 0.95):
            cv2.putText(frame, 'Durmiendo : ' + str(round(list(prediction)[0][1], 3)), 
                        (10, int(height * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            return 1  # Dormido

        if (list(prediction)[0][0] >= 0.95):
            cv2.putText(frame, 'Despierto', (10, int(height * 0.08)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            return 0  # Despierto

    return None

def generate_frames():
    global contador
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        # Predicción de sueño
        state = predict_sleep(frame, height, width)

        # Controlar el reproductor de audio basado en la predicción
        if state == 1:
            contador += 1
            if contador >= 15:
                contador = 15
                if not p.is_playing():
                    p.play()
        elif state == 0:
            contador -= 1
            if contador <= 0:
                contador = 0
                p.stop()

        # Convertir la imagen a formato adecuado para enviar al navegador
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # Usamos threaded=True para mejorar el rendimiento
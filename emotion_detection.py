import cv2
from PIL import Image
from ultralytics import YOLO
# from audio_whisper import AudioTranscriber

model_face    = YOLO("YOLO11_10B_face.pt")
model_emotion = YOLO("YOLO11_20B_emotion.pt")

cap = cv2.VideoCapture(0)

print("[DEBUG] Importando AudioTranscriber...")
from audio_whisper import AudioTranscriber
print("[DEBUG] Creando objeto AudioTranscriber...")

audio = AudioTranscriber(model_name="small.en")

print("[DEBUG] Llamando audio.start()...")
audio.start()
print("[INFO] Todo iniciado correctamente")



try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_results = model_face(frame)

        for face_res in face_results:
            for box in face_res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[y1:y2, x1:x2]

                emotion_results = model_emotion(face_crop)
                em_box = emotion_results[0].boxes[0]
                label = emotion_results[0].names[int(em_box.cls)]
                conf  = float(em_box.conf)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}',
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0,255,0), 2)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Libera recursos
    audio.stop()  
    cap.release()
    cv2.destroyAllWindows()

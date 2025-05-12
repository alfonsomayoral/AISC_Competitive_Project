import os, csv, threading, queue, numpy as np, sounddevice as sd, webrtcvad
import whisper
import time

SAMPLE_RATE = 16000
CHUNK_MS    = 30
VAD_LEVEL   = 2

class AudioTranscriber:
    def __init__(self, model_name="small.en"):
        print("[AUDIO] Cargando modelo Whisper...")
        try:
            self.model = whisper.load_model(model_name)
        except Exception as e:
            print(f"[ERROR] Error al cargar Whisper: {e}")
            raise

        try:
            self.vad = webrtcvad.Vad(VAD_LEVEL)
        except Exception as e:
            print(f"[ERROR] Error con WebRTC VAD: {e}")
            raise

        self.q      = queue.Queue()
        self.buffer = bytes()
        self._start_ts = time.time()
        self._stop  = threading.Event()

        # Preparar CSV
        os.makedirs("data", exist_ok=True)
        self.out_path = "data/transcripts.csv"
        with open(self.out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp_s", "text"])

        print("[AUDIO] Whisper y VAD cargados.")

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[AUDIO][WARN] InputStream status: {status}")
        self.q.put(indata.copy())

    def start_stream(self):
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='int16',
                blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
                callback=self._callback
            )
            self.stream.start()
            print("[AUDIO] Grabación de audio iniciada.")
        except Exception as e:
            print(f"[ERROR] No se pudo iniciar el micrófono: {e}")
            raise

    def _run(self):
        while not self._stop.is_set():
            if not self.q.empty():
                chunk = self.q.get()
                pcm = chunk.tobytes()
                if self.vad.is_speech(pcm, SAMPLE_RATE):
                    self.buffer += pcm
            if len(self.buffer) / 2 / SAMPLE_RATE > 5:
                self._transcribe_buffer()
                self.buffer = bytes()
            time.sleep(0.01)

    def _transcribe_buffer(self):
        try:
            audio_np = np.frombuffer(self.buffer, np.int16).astype(np.float32) / 32768.0
            segments = self.model.transcribe(audio_np, language="en")["segments"]

            now_ts = time.time() - self._start_ts  # tiempo relativo desde inicio

            with open(self.out_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for seg in segments:
                    writer.writerow([f"{now_ts:.2f}", seg["text"].strip()])
        except Exception as e:
            print(f"[ERROR] Error durante la transcripción: {e}")

    def start(self):
        self.start_stream()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        print("[AUDIO] Deteniendo audio...")
        self._stop.set()
        self.thread.join()
        if hasattr(self, "stream"):
            self.stream.stop()
            self.stream.close()
        if self.buffer:
            self._transcribe_buffer()
        print("[AUDIO] Transcripción finalizada y guardada.")
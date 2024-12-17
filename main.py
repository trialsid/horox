import os
from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import threading
import re
from datetime import datetime
from flask_socketio import SocketIO, emit
import sounddevice as sd
import numpy as np
import queue

load_dotenv()

app = Flask(__name__)

socketio = SocketIO(app)

# Audio configuration
CHANNELS = 1
RATE = 44100
CHUNK = 1024
audio_queue = queue.Queue()

audio_stream = None
selected_device = None

def audio_callback(indata, frames, time, status):
    """This is called for each audio block"""
    if status:
        print(status)
    audio_queue.put(indata.copy())

def audio_sender():
    """Send audio data to connected clients"""
    while True:
        try:
            data = audio_queue.get()
            audio_data = data.tobytes()
            socketio.emit('audio_stream', {'audio': audio_data})
        except Exception as e:
            print(f"Error in audio sender: {e}")

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def start_streaming():
    """Start the audio stream"""
    global audio_stream, selected_device
    try:
        # List available audio devices
        devices = sd.query_devices()
        print("\nAvailable audio devices:")
        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']}")
        
        # Get user input for device selection
        device_index = input("\nEnter the number of your Bluetooth microphone device: ")
        selected_device = int(device_index)
        
        print(f"\nStarting audio stream with device {selected_device}...")
        
        audio_stream = sd.InputStream(
            device=selected_device,
            channels=CHANNELS,
            samplerate=RATE,
            callback=audio_callback,
            blocksize=CHUNK
        )
        audio_stream.start()
        print("Audio streaming started successfully")
        return True
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        return False

# Start audio sender thread
sender_thread = threading.Thread(target=audio_sender)
sender_thread.daemon = True
sender_thread.start()

# --- Gemini Setup ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: GEMINI_API_KEY environment variable not set.")
    exit()

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

speech_status = {"completed": False, "startTime": None, "endTime": None}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="I feed the generated text to a TTS model. So don't include extra precursor text. Direct horoscopy. Be practical and cautious in the advice. No subheading or subtitles. Just paragraphs. Users are rural Telugu people doing various occupations. Write like a screenplay writer. Discuss past, present and future subtly. End with \"మల్దకల్ Thimmappa Swamy blessings on you\"",
)

# --- Azure Speech Setup ---
try:
  speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
  audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
  speech_config.speech_synthesis_voice_name = 'te-IN-MohanNeural'  # Default Telugu voice
except KeyError:
    print("Error: SPEECH_KEY or SPEECH_REGION environment variables not set.")
    exit()


def construct_prompt(name, place_of_birth, dob, problem, occupation):
    """Constructs the initial prompt based on user input."""

    prompt_parts = [
        f"Name: {name}",
        f"Place of Birth: {place_of_birth}",
        f"DOB(mm-yyyy): {dob}",
    ]
    if problem:
        prompt_parts.append(f"Problem: {problem}")
    if occupation:
        prompt_parts.append(f"Occupation: {occupation}")

    prompt_parts.append(
        "\n\nBased on the above information of a person write positive uplifting horoscopy for the user. Write only in Telugu. Include deity references. Advise you give should be practical and progressive. Give negatives and positives concerning the life situation. Mention his horoscope. Go out of the box and give solutions. Write very very short. Compress."
    )
    return "\n".join(prompt_parts)


def get_horoscope(initial_prompt):
    """Generates the horoscope based on the constructed prompt."""
    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [initial_prompt]}]
    )
    try:
        response = chat_session.send_message(content=".")
        return response.text.replace("\n", " ")
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None


def speak_text(text, completion_callback):
    global speech_status
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    
    def synthesis_started(evt):
        global speech_status
        start_time = datetime.now()
        speech_status["startTime"] = start_time
        speech_status["completed"] = False
        print(f"Speech synthesis started at {start_time}")

    def synthesis_completed(evt):
        global speech_status
        end_time = datetime.now()
        speech_status["endTime"] = end_time
        speech_status["completed"] = True  # Mark as completed
        print(f"Speech synthesis completed at {end_time}")
        completion_callback(True, speech_status["startTime"], end_time, None)

    speech_synthesizer.synthesis_started.connect(synthesis_started)
    speech_synthesizer.synthesis_completed.connect(synthesis_completed)

    try:
        speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
        
        if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Speech synthesis completed successfully")
        elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
            print(f"Speech synthesis canceled: {speech_synthesis_result.cancellation_details.reason}")
            speech_status["completed"] = True  # Mark as completed even on cancellation
            completion_callback(False, None, None, "Speech synthesis canceled")
    except Exception as e:
        print(f"Exception during speech synthesis: {e}")
        speech_status["completed"] = True  # Mark as completed on exception
        completion_callback(False, None, None, str(e))
def on_completion(success, start_time, end_time, error_message):
    response = {
        "success": success,
        "startTime": start_time.isoformat() if start_time else None,
        "endTime": end_time.isoformat() if end_time else None,
        "error": error_message
    }
    # Return the response directly rather than wrapping in jsonify
    return response

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form.get("name")
        place_of_birth = request.form.get("place_of_birth")
        dob = request.form.get("dob")
        problem = request.form.get("problem")
        occupation = request.form.get("occupation")

        # Basic server-side validation
        errors = {}
        if not name:
            errors['name'] = 'Name is required.'
        if not place_of_birth:
            errors['place_of_birth'] = 'Place of birth is required.'
        if not dob:
           errors['dob'] = 'Date of birth is required.'
        elif not re.match(r'^(0[1-9]|1[0-2])-\d{4}$', dob):
            errors['dob'] = 'Invalid date format. Use mm-yyyy.'

        if errors:
            return jsonify({'errors': errors}), 400

        initial_prompt = construct_prompt(name, place_of_birth, dob, problem, occupation)
        horoscope_text = get_horoscope(initial_prompt)

        if horoscope_text is None:
            return jsonify({'error': 'Failed to generate horoscope text.'}), 500
        print("Horoscope Text:", horoscope_text)

        # Call speak_text in a new thread with the callback
        threading.Thread(target=speak_text, args=(horoscope_text, lambda success,start,end,error:  on_completion(success,start,end,error))).start()
        return jsonify({
            "message": "Generating horoscope...",
            "fullText": horoscope_text
        })
    return render_template("index.html")

@app.route("/speech_status")
def get_speech_status():
    return jsonify(speech_status)

@app.route("/start_audio", methods=["POST"])
def start_audio():
    global audio_stream
    if audio_stream is None:
        success = start_streaming()
        if success:
            return jsonify({"status": "success", "message": "Audio stream started"})
        return jsonify({"status": "error", "message": "Failed to start audio stream"}), 500
    return jsonify({"status": "error", "message": "Audio stream already running"}), 400

if __name__ == "__main__":
    # Start audio sender thread
    sender_thread = threading.Thread(target=audio_sender)
    sender_thread.daemon = True
    sender_thread.start()
    
    # Run the Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
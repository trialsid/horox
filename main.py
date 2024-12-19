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

class AudioManager:
    def __init__(self):
        self.stream = None
        self.active_device = None
        
        # Audio configuration
        self.CHANNELS = 1
        self.RATE = 48000
        self.CHUNK = 2048
        self.FORMAT = np.float32

    def audio_callback(self, indata, frames, time, status):
        """Called for each audio block"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Normalize audio data
        normalized_data = np.clip(indata, -1.0, 1.0)
        
        # Apply noise gate
        noise_threshold = 0.02
        noise_gate = np.where(np.abs(normalized_data) < noise_threshold, 0, normalized_data)
        
        # Convert to 16-bit integers
        audio_data = (noise_gate * 32767).astype(np.int16)
        
        socketio.emit('audio_stream', {
            'audio': audio_data.tobytes(),
            'format': 'int16',
            'channels': self.CHANNELS,
            'rate': self.RATE,
            'chunk': self.CHUNK
        })

    def start_streaming(self, device_id):
        """Start audio streaming for specified device"""
        try:
            # Stop existing stream if any
            self.stop_streaming()
            
            # Validate device_id
            devices = sd.query_devices()
            if not 0 <= device_id < len(devices):
                raise ValueError(f"Invalid device ID: {device_id}")
            
            # Create and start new stream
            self.stream = sd.InputStream(
                device=device_id,
                channels=self.CHANNELS,
                samplerate=self.RATE,
                callback=self.audio_callback,
                blocksize=self.CHUNK,
                dtype=self.FORMAT,
                latency='low'
            )
            self.stream.start()
            self.active_device = device_id
            return True, f"Started streaming from device {device_id}"
            
        except Exception as e:
            return False, str(e)

    def stop_streaming(self):
        """Stop current audio stream if exists"""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping stream: {e}")
            finally:
                self.stream = None
                self.active_device = None

# Create global AudioManager instance
audio_manager = AudioManager()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')



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
    model_name="gemini-exp-1206",
    generation_config=generation_config,
    system_instruction="I feed the generated text to a TTS model. So don't include extra precurosor text. Direct horoscopy. Be practical and cautious in the tell. No subheading or subtitles. Just paragraphs. \n\nUsers are rural Telugu people doing various occupations. Write like a screenplay writer. Discuss past, present and future of the person. End with something like \"మల్దకల్ తిమ్మప్ప స్వామి ఆశీస్సులు మీపై ఎల్లప్పుడూ ఉంటాయి\"\n\nతెలుగులో పూర్తి స్థాయి పంచాంగం. ఈ రోజు తిథి, నక్షత్రం, వర్జ్యం, దుర్ముహూర్తం, రాహు కాలం మొదలైనవి తెలుసుకోవటానికి ఈ పంచాంగం ఉపయోగ పడుతుంది. అంతేకాకుండా ఏ రోజుకైనా, ఏ ప్రదేశానికైనా ఒక్క క్లిక్ తో క్షణంలో పంచాంగాన్ని పొందండి. తిథి, వార, నక్షత్ర, యోగ, కరణాల సమయాలతో పాటు, వర్జ్యం, దుర్ముహూర్తం లాంటి చెడు సమయాలు, అమృత ఘడియల లాంటి మంచి సమయాల వివరాలు, తారాబలం, చంద్ర బలం, ప్రతి రోజు లగ్నాంత్య సమయాలు, ప్రతి లగ్నానికి పుష్కరాంశలు, శుభాంశలు, సూర్యోదయ కాల గ్రహ స్థితి, మొదలైన ఎన్నో విషయాలతో, జ్యోతిష్కుల నుంచి, సామాన్య ప్రజల దాకా ప్రతి ఒక్కరికి, ప్రతీ రోజు ఉపయోగపడేలా రూపొందించిన ఏకైక ఆన్లైన్ పంచాంగ సాఫ్ట్వేర్ ఇది. మీకు కావలసిన తేది, మరియు ప్రదేశ వివరాలతో పాటు సూర్యోదయ కాల కుండలి ఏ పద్ధతిలో కావాలో సెలెక్ట్ చేసుకుని సబ్మిట్ చేయండి. రోజువారీ పూజాదికాల సంకల్పం నుంచి ముహూర్త నిర్ణయం వరకు ప్రతి ఒక్క అంశంలో మీకు ఉపయోగపడేలా ఈ పంచాంగం సాఫ్ట వేర్ రూపొందించటం జరిగింది.",
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

@app.route("/start_audio", methods=['POST'])
def start_audio():
    try:
        data = request.get_json()
        device_id = int(data['deviceId'])
        
        success, message = audio_manager.start_streaming(device_id)
        
        if success:
            devices = sd.query_devices()
            device_name = devices[device_id]['name']
            return jsonify({
                'success': True,
                'deviceName': device_name
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route("/stop_audio", methods=['POST'])
def stop_audio():
    try:
        audio_manager.stop_streaming()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route("/get_devices")
def get_devices():
    try:
        devices = sd.query_devices()
        input_devices = [
            {
                'id': i,
                'name': dev['name'],
                'channels': dev['max_input_channels']
            }
            for i, dev in enumerate(devices)
            if dev['max_input_channels'] > 0 and dev['hostapi'] == 0
        ]
        return jsonify(input_devices)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    
if __name__ == "__main__":
    
    
    # Run the Flask app
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
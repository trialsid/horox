import os
import google.generativeai as genai
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

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

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction="I feed the generated text to a TTS model. So don't include extra precursor text. Direct horoscopy. Be practical and cautious in the advice. No subheading or subtitles. Just paragraphs. Users are rural Telugu people doing various occupations. Write like a screenplay writer. Discuss past, present and future subtly. End with \"మల్దకల్ Thimmappa Swamy blessings on you\"",
)

def get_user_input():
    """Gets user details from the command line."""
    name = input("Enter your name: ")
    place_of_birth = input("Enter your place of birth: ")
    dob = input("Enter your DOB (mm-yyyy): ")

    problem = input("Enter your problem (optional, press Enter to skip): ")
    occupation = input("Enter your occupation (optional, press Enter to skip): ")

    return name, place_of_birth, dob, problem, occupation


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
        "\nBased on the above information of a person write positive uplifting horoscopy for the user. Write only in Telugu. Include deity references. Advise you give should be practical and progressive. Give negatives and positives concerning the life situation. Mention his horoscope. Go out of the box and give solutions. Write long text."
    )
    return "\n".join(prompt_parts)


def get_horoscope(initial_prompt):
    """Generates the horoscope based on the constructed prompt."""
    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [initial_prompt]}]
    )
    try:
      response = chat_session.send_message(content=".")
      return response.text.replace("\n", " ")  # Remove newline characters here
    except Exception as e:
        return f"Error: {e}"

# --- Azure Speech Setup ---
def speak_text(text, speech_config, audio_config):
    """Synthesizes text to speech and handles the result."""
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text: {}".format(text))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
                print("Did you set the speech resource key and region values?")


def main():
    # Configure speech settings
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_config.speech_synthesis_voice_name = 'te-IN-MohanNeural'  # Default Telugu voice

    name, place_of_birth, dob, problem, occupation = get_user_input()
    initial_prompt = construct_prompt(name, place_of_birth, dob, problem, occupation)
    horoscope_text = get_horoscope(initial_prompt)
    print("Horoscope Text:", horoscope_text)

    speak_text(horoscope_text, speech_config, audio_config)


if __name__ == "__main__":
    main()
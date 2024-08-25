import sounddevice as sd
import numpy as np
import time
import wave
from threading import Thread
import os
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI as OpenAIApi
from IPython.display import Audio
from io import BytesIO
from langchain_openai import OpenAI as LangchainOpenAI
import streamlit as st
import nltk

your_open_api_key = "sk-kJi_JWeyVE344i5QNg_GTw5YEvSq3183Bx74Wv_rVdT3BlbkFJLlp8i0jL3HYvS-vWhUt5BUuCTnQ9tbp8vjhYZCTK4A"
os.environ['OPENAI_API_KEY'] = your_open_api_key
st.title("Voice Based Real Time Seller")

CHUNK = 1024
FORMAT = 'int16'
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 500  # Threshold for detecting silence
SILENCE_DURATION = 2  # Duration to wait for silence in seconds

class AudioRecorder(Thread):
    def __init__(self):
        super().__init__()
        self.recording = True
        self.audio_data = []

    def run(self):
        def callback(indata, frames, time, status):
            audio_np = np.abs(indata)
            if np.max(audio_np) < SILENCE_THRESHOLD:
                if time.time() - self.start_time > SILENCE_DURATION:
                    self.recording = False
            else:
                self.start_time = time.time()

            if self.recording:
                self.audio_data.append(indata.copy())

        self.start_time = time.time()
        print("Recording...")
        with sd.InputStream(channels=CHANNELS, callback=callback, dtype=FORMAT, samplerate=RATE):
            while self.recording:
                time.sleep(0.1)
        print("Recording stopped.")

prompt = f"""
You are a friendly and knowledgeable sales assistant specializing in fashion.
Give the output in human English like a real person.
Respond naturally and conversationally, as if you were a real person.
Provide personalized clothing recommendations, focusing on style and fit.
Keep your answers short, sweet, and helpful, without unnecessary details like product ids unless the customer asks for them.

Customer: {{input}}
"""

# Initialize the OpenAI model
openai_llm = LangchainOpenAI()

# Initialize memory to keep track of chat history
memory = ConversationBufferMemory()

# Initialize the SQL database connection
input_db = SQLDatabase.from_uri("sqlite:///Database.db")

# Use the LLM instance in the SQLDatabaseChain
db_chain = SQLDatabaseChain.from_llm(llm=openai_llm, db=input_db, verbose=False)

def get_response(user_input):
    # Combine the system prompt with the current chat history
    full_prompt = f"{prompt}\n{memory.chat_memory.messages}\nCustomer: {user_input}"

    # Run the SQLDatabaseChain with the given prompt
    response = db_chain.run(full_prompt)

    # Update memory with the latest interaction
    memory.chat_memory.add_message(f"Customer: {user_input}")  # Combine sender and message into a single string
    memory.chat_memory.add_message(f"Seller: {response}")  # Combine sender and message into a single string

    return response

process_url_clicked = st.sidebar.button("click here to start recording")
main_placefolder = st.empty()
if process_url_clicked:
    # Start recording
    recorder = AudioRecorder()
    recorder.start()

    # Wait for recording to finish
    recorder.join()

    # Save the audio data to a WAV file
    audio_data = np.concatenate(recorder.audio_data)
    with wave.open('output2.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(np.dtype(FORMAT).itemsize)
        wf.setframerate(RATE)
        wf.writeframes(audio_data.tobytes())
    
    st.success("Recording saved successfully.")
    client = OpenAIApi()
    audio_file = open("output2.wav", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    input = transcription.text
    main_placefolder.text("Processing the data...")
    text = get_response(input)
    main_placefolder.text("Generating the responses...")
    response = client.audio.speech.create(
        model="tts-1-hd",
        voice="alloy",
        input=text,
    )

    audio_data = response.content
    audio_file = BytesIO(audio_data)
    st.audio(audio_file, format='audio/wav')
    st.markdown(
        """
        <script>
        const audio = document.querySelector('audio');
        if (audio) {
            audio.play();
        }
        </script>
        """,
        unsafe_allow_html=True
    )

    # Use IPython.display.Audio to play the audio in Jupyter/Colab
    Audio(data=audio_data, autoplay=True)
    main_placefolder.text(f"response:{text}")

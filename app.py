import pyaudio
import numpy as np
import time
import wave
from threading import Thread
import os
import sqlite3
from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI as OpenAIApi
from IPython.display import Audio
from io import BytesIO
from langchain import OpenAI as LangchainOpenAI
import streamlit as st
import nltk



os.environ['OPENAI_API_KEY'] = your_open_api_key
st.title("Voice Based Real Time Seller")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SILENCE_THRESHOLD = 500  # Threshold for detecting silence
SILENCE_DURATION = 5  # Duration to wait for silence in seconds

class AudioRecorder(Thread):
    def __init__(self):
        super().__init__()
        self.recording = True
        self.audio_data = []

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

        print("Recording...")

        start_time = time.time()
        while self.recording:
            data = stream.read(CHUNK)
            self.audio_data.append(data)
            audio_np = np.frombuffer(data, dtype=np.int16)
            if np.max(np.abs(audio_np)) < SILENCE_THRESHOLD:
                if time.time() - start_time > SILENCE_DURATION:
                    self.recording = False
            else:
                start_time = time.time()

        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Recording stopped.")

prompt = f"""
 You are a friendly and knowledgeable sales assistant specializing in fashion.
 give the output in the human english like a real person.
 Respond naturally and conversationally, as if you were a real person.
 Provide personalized clothing recommendations, focusing on style and fit.
 Keep your answers short, sweet, and helpful, without unnecessary details like(product ids ), if the coustmer were asking the give it.


 Customer: {input}

  """

# Initialize the OpenAI model
openai_llm = LangchainOpenAI()

# Initialize memory to keep track of chat history
memory = ConversationBufferMemory()

# Initialize the SQL database connection
input_db = SQLDatabase.from_uri("sqlite:///your_database.db")

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
process_url_clicked= st.sidebar.button("click here to start recording")
main_placefolder=st.empty()
if process_url_clicked:
    # Start recording
    recorder = AudioRecorder()
    recorder.start()

# Wait for recording to finish
    recorder.join()

# Save the audio data to a WAV file
    with wave.open('output2.wav', 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(recorder.audio_data))
    st.success("Recording saved successfully.")
    client = OpenAIApi()
    audio_file= open("a.wav", "rb")
    transcription = client.audio.translations.create(
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
          input= text,
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




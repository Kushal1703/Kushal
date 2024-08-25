
# Voice-Based Fashion Sales Assistant

This project aims to build a voice-based sales assistant specializing in fashion. The assistant leverages advanced AI models to provide personalized clothing recommendations through natural language conversation.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)

## Introduction

The Voice-Based Fashion Sales Assistant is designed to assist users in making fashion choices by providing real-time recommendations through voice interaction. The system uses a combination of audio processing and natural language processing (NLP) techniques to understand and respond to user queries, making the shopping experience more interactive and personalized.

## Features

- **Voice Interaction**: Users can interact with the assistant using voice commands.
- **AI-Powered Recommendations**: Utilizes the OpenAI and Langchain models to offer personalized fashion advice.
- **Database Integration**: Connects to a SQL database to retrieve and recommend products based on user preferences.
- **Real-Time Responses**: Provides immediate feedback and recommendations in natural language.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Kushal1703/voice-fashion-assistant.git
    cd voice-fashion-assistant
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have access to the necessary API keys for OpenAI and other services.

## Usage

1. Run the setup script to install the necessary packages:
    ```bash
    sudo apt-get install sqlite3
    pip install pyaudio langchain langchain_community openai langchain_experimental pydub simpleaudio sounddevice soundfile
    ```

2. Start the voice assistant by running the main script:
    ```bash
    python voice_sellers_model.ipynb
    ```

3. Speak your fashion-related query, and the assistant will respond with personalized recommendations.

Certainly! Here's an improved version of the **Methodology** section for your README file:

---

## Methodology

This project integrates various advanced technologies to create a voice-based fashion sales assistant that delivers real-time, personalized recommendations. The methodology involves several key components:

### 1. **Audio Capture and Processing**

- **Audio Input**: The system utilizes the `pyaudio` library to capture audio input from the user in real-time. The audio is processed in chunks to ensure smooth and continuous interaction.
- **Silence Detection**: A silence detection mechanism is implemented to determine when the user has finished speaking, based on a predefined silence threshold. This helps in segmenting the user's input and initiating the response generation process without unnecessary delays.

### 2. **Natural Language Processing (NLP)**

- **Speech-to-Text Conversion**: Once the audio is captured, it is converted into text using a speech recognition engine. This text serves as the input for generating fashion recommendations.
- **Conversational AI**: The converted text is processed using `Langchain` and `OpenAI` models, which have been fine-tuned to understand and respond to fashion-related queries. The assistant uses a predefined system prompt to maintain a conversational tone, ensuring that responses are natural, context-aware, and customer-friendly.

### 3. **Database Interaction**

- **SQL Database Integration**: The system is integrated with an SQLite database that stores detailed information about various fashion products. This includes categories, styles, sizes, colors, and other relevant attributes.
- **SQL Queries**: The assistant generates SQL queries dynamically based on user input to retrieve the most relevant product information from the database. This allows the system to recommend items that closely match the user's preferences and needs.

### 4. **Personalized Recommendations**

- **Feature Matching**: The system matches the user’s input with available products by comparing features such as style, fit, and color. This is done by leveraging the extracted information from the NLP models and database queries.
- **Response Generation**: After determining the most suitable products, the system generates a personalized recommendation in natural language. The response is crafted to be concise, informative, and aligned with the user's query.

### 5. **Text-to-Speech Conversion**

- **Audio Feedback**: The generated response is converted back into audio using a text-to-speech (TTS) engine. The assistant uses a high-quality, human-like voice to deliver the recommendations, ensuring a seamless and engaging user experience.
- **Playback**: The audio response is then played back to the user, completing the interaction loop.

### 6. **Memory Management**

- **Conversation Tracking**: The system uses `ConversationBufferMemory` to keep track of the ongoing conversation, allowing it to provide contextually relevant responses throughout the session. This ensures that the assistant remembers previous interactions and can build on them, offering a more coherent and personalized experience.

---

This enhanced methodology section provides a more detailed and structured explanation of how the voice-based fashion sales assistant works, highlighting the integration of multiple technologies and the process flow from user input to final audio output.

## Results

The assistant provides real-time voice responses with fashion recommendations. It interacts naturally and conversationally, enhancing the user experience during the shopping process.

---

This README should help provide a clear overview of the project, its setup, and how to use it. If you have any specific details you'd like to include or modify, feel free to let me know!

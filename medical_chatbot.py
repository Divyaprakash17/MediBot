# ============================================
# IMPORT SECTION
# ============================================
import os
import json
import time
import threading
import streamlit as st
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# ============================================
# VOICE PROCESSING SETUP
# ============================================
# Try to import voice-related modules
VOICE_AVAILABLE = True
try:
    import speech_recognition as sr
    import pyttsx3
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    from scipy.io.wavfile import write
    import tempfile
except ImportError as e:
    VOICE_AVAILABLE = False
    st.warning(f"Voice features are not available. Some dependencies are missing: {e}")
    import warnings
    warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure Gemini API
try:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if not GEMINI_API_KEY:
        st.error("Please set the GEMINI_API_KEY environment variable in the .env file")
        st.stop()
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    GEMINI_AVAILABLE = True
except Exception as e:
    st.error(f"Error initializing Gemini: {str(e)}")
    st.warning("Running in test mode with limited functionality. Please check your API key and internet connection.")
    GEMINI_AVAILABLE = False

# Initialize speech recognizer and TTS engine if available
if VOICE_AVAILABLE:
    recognizer = sr.Recognizer()
    tts_engine = pyttsx3.init()

# Set up data directory
DATA_DIR = Path("chat_history")
DATA_DIR.mkdir(exist_ok=True)

# ============================================
# UTILITY FUNCTIONS
# ============================================
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ============================================
# TEXT-TO-SPEECH (TTS) IMPLEMENTATION
# ============================================
# Windows TTS implementation using win32com
import win32com.client
import queue
import time

# Global TTS engine and state
_tts_engine = None
_tts_lock = threading.Lock()
_speech_queue = queue.Queue()
_is_speaking = False
VOICE_AVAILABLE = True

class TTSThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        
    def run(self):
        global _is_speaking
        
        while not self._stop_event.is_set():
            try:
                # Wait for a short time to prevent busy waiting
                time.sleep(0.1)
                
                if not _is_speaking and not _speech_queue.empty():
                    _is_speaking = True
                    text = _speech_queue.get()
                    
                    try:
                        if _tts_engine is not None:
                            print(f"Speaking: {text}")
                            _tts_engine.Speak(text)
                    except Exception as e:
                        print(f"Error in speech synthesis: {e}")
                    finally:
                        _is_speaking = False
                        _speech_queue.task_done()
                        
            except Exception as e:
                print(f"Error in TTS thread: {e}")
    
    def stop(self):
        self._stop_event.set()

# ============================================
# TTS INITIALIZATION
# ============================================
# Initialize TTS engine and thread
_tts_thread = None
_tts_initialized = False

def init_tts_engine():
    """Initialize the Windows TTS engine"""
    global _tts_engine, _tts_thread, VOICE_AVAILABLE, _tts_initialized
    
    if _tts_initialized:
        return _tts_engine
        
    try:
        with _tts_lock:
            if _tts_engine is None:
                _tts_engine = win32com.client.Dispatch("SAPI.SpVoice")
                print("Windows TTS engine initialized successfully")
                
                # List available voices
                voices = _tts_engine.GetVoices()
                voice_names = [voice.GetDescription() for voice in voices]
                print(f"Available voices: {voice_names}")
                
                # Try to find a female voice (Zira is the default female voice on Windows)
                for i, voice in enumerate(voices):
                    if 'Zira' in voice.GetDescription():
                        _tts_engine.Voice = voices.Item(i)
                        print(f"Using voice: {voice.GetDescription()}")
                        break
                
                # Set rate (-10 to 10, 0 is normal)
                _tts_engine.Rate = 0
                
                # Start the TTS thread if not already running
                if _tts_thread is None or not _tts_thread.is_alive():
                    _tts_thread = TTSThread()
                    _tts_thread.start()
            
            _tts_initialized = True
            return _tts_engine
            
    except Exception as e:
        print(f"Failed to initialize Windows TTS engine: {e}")
        VOICE_AVAILABLE = False
        return None

def speak(text):
    """Add text to the speech queue"""
    global VOICE_AVAILABLE
    
    if not VOICE_AVAILABLE or not text.strip():
        return
    
    # Initialize TTS engine if not already done
    if _tts_engine is None:
        init_tts_engine()
    
    # Add text to the queue
    try:
        _speech_queue.put(text)
    except Exception as e:
        print(f"Error adding to speech queue: {e}")

def listen():
    """Listen to user's voice input and convert to text"""
    global VOICE_AVAILABLE
    
    if not VOICE_AVAILABLE:
        st.warning("Voice functionality is not available. Please check your microphone and audio settings.")
        return ""
    
    try:
        import io
        import wave
        
        # Audio recording parameters
        fs = 16000  # Sample rate
        seconds = 5  # Recording duration
        
        st.info("Listening... Speak now (5 seconds)")
        st.session_state.listening = True
        
        try:
            # List available input devices
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            st.sidebar.info(f"Using audio input device: {devices[default_input]['name']}")
            
            # Record audio directly to memory
            try:
                recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
                sd.wait()  # Wait until recording is finished
                
                # Check if recording is silent
                if np.abs(recording).max() < 100:  # Threshold for silence
                    st.warning("No audio detected. Please check your microphone.")
                    return ""
                
            except Exception as e:
                st.error(f"Error recording audio: {str(e)}")
                st.warning("Please ensure your microphone is properly connected and not in use by another application.")
                return ""
            
            # Create an in-memory WAV file
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(fs)
                wf.writeframes(recording.tobytes())
            
            # Reset buffer position
            wav_io.seek(0)
            
            # Convert to audio data for speech recognition
            r = sr.Recognizer()
            with sr.AudioFile(wav_io) as source:
                try:
                    # Adjust for ambient noise with shorter timeout
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = r.record(source)
                    st.info("Processing your voice...")
                    
                    try:
                        # Try Google Speech Recognition first
                        text = r.recognize_google(audio_data, language="en-US")
                        st.success("Speech recognized successfully!")
                        return text
                    except sr.UnknownValueError:
                        st.warning("Could not understand audio. Please speak clearly.")
                        return ""
                    except sr.RequestError as e:
                        st.warning(f"Could not request results from Google Speech Recognition service; {e}")
                        try:
                            # Fall back to Sphinx if Google fails
                            text = r.recognize_sphinx(audio_data, language="en-US")
                            st.success("Speech recognized using offline mode!")
                            return text
                        except Exception as e:
                            st.warning(f"Offline recognition failed: {e}")
                            return ""
                            
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                    return ""
                    
        except Exception as e:
            st.error(f"Error during recording: {str(e)}")
            return ""
            
        finally:
            st.session_state.listening = False
            
    except Exception as e:
        st.error(f"Unexpected error in voice input: {str(e)}")
        VOICE_AVAILABLE = False
        st.error("Voice input has been disabled due to an error. Please refresh the page to try again.")
        return ""

def should_generate_summary():
    """Determine if we should generate a summary after sufficient questions"""
    if 'conversation' not in st.session_state:
        return False
        
    # Count number of user messages (questions answered)
    user_msgs = [m for m in st.session_state.conversation if m['role'] == 'user']
    num_questions = len(user_msgs)
    
    # Generate summary after 7-8 questions and haven't given a summary yet
    return num_questions >= 7 and not st.session_state.get('summary_given', False)

# ============================================
# CONVERSATION LOGIC
# ============================================
def get_next_question(conversation_history, user_input):
    """Generate next question using LLM based on conversation history"""
    try:
        # Initialize or update conversation state
        if 'symptom_summary' not in st.session_state:
            st.session_state.symptom_summary = {}
        if 'asked_questions' not in st.session_state:
            st.session_state.asked_questions = set()
        if 'last_question' not in st.session_state:
            st.session_state.last_question = ""
            
        # Check if we should generate a summary after enough questions
        if should_generate_summary():
            st.session_state.summary_given = True
            return generate_final_recommendation()
        
        # Update symptom summary from conversation
        user_input_lower = user_input.lower()
        
        # Track symptoms and their details
        symptom_keywords = {
            'fever': ['fever', 'temperature'],
            'headache': ['headache', 'head pain', 'head ache'],
            'nausea': ['nausea', 'nauseous', 'nauseated'],
            'vomiting': ['vomit', 'threw up', 'throwing up'],
            'dizziness': ['dizzy', 'lightheaded'],
            'fatigue': ['tired', 'fatigue', 'exhausted']
        }
        
        # Update symptom summary with latest information
        for symptom, keywords in symptom_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                if symptom not in st.session_state.symptom_summary:
                    st.session_state.symptom_summary[symptom] = {
                        'mentioned': True,
                        'details': user_input,
                        'severity': None,
                        'duration': None
                    }
                else:
                    # Update existing symptom with more details
                    st.session_state.symptom_summary[symptom]['details'] = user_input
        
        # Prepare the conversation context for the LLM
        prompt_parts = [
            """You are a helpful medical assistant conducting a patient interview. 
            Your goal is to gather relevant medical information by asking one clear, focused question at a time.
            Keep questions concise and easy to understand. Focus on medical symptoms, history, and relevant details.
            
            IMPORTANT RULES:
            1. NEVER repeat questions that have already been asked
            2. If a symptom is mentioned, ask follow-up questions about it before moving on
            3. If you need to ask about a symptom that was mentioned, reference the previous mention
            4. Keep questions focused and ask one thing at a time
            5. If the patient's response is unclear, ask for clarification
            6. When you have enough information, provide a brief assessment
            7. If you've asked about all symptoms, provide a summary and next steps\n\n"""
        ]
        
        # Add context about the current conversation state
        prompt_parts.append("CURRENT CONVERSATION STATE:")
        if st.session_state.symptom_summary:
            prompt_parts.append("Symptoms mentioned:")
            for symptom, details in st.session_state.symptom_summary.items():
                prompt_parts.append(f"- {symptom.capitalize()}: {details.get('details', 'No details')}")
        else:
            prompt_parts.append("No specific symptoms mentioned yet.")
            
        if st.session_state.asked_questions:
            prompt_parts.append("\nQuestions already asked:")
            for q in st.session_state.asked_questions:
                prompt_parts.append(f"- {q}")
        
        prompt_parts.append("\nCONVERSATION HISTORY:")
        
        # Add symptom summary if available
        if st.session_state.symptom_summary:
            prompt_parts.append("CURRENT SYMPTOM SUMMARY:")
            for symptom, details in st.session_state.symptom_summary.items():
                prompt_parts.append(f"- {symptom.capitalize()}: {details}")
            prompt_parts.append("")
        
        # Add conversation history (limit to last 10 exchanges for context)
        recent_history = st.session_state.conversation[-10:]
        for msg in recent_history:
            role = "Patient" if msg["role"] == "user" else "Assistant"
            content = msg['content'].strip()
            if content:
                # Add question to asked questions if it's from the assistant
                if role == "Assistant" and '?' in content and content not in st.session_state.asked_questions:
                    st.session_state.asked_questions.add(content.split('?')[0] + '?')
                prompt_parts.append(f"{role}: {content}")
        
        # Add the latest user input if not already in conversation
        if user_input.strip():
            prompt_parts.append(f"Patient: {user_input}")
        
        # Add instruction for the next response
        prompt_parts.append("\nBased on the conversation so far, what is the most appropriate question to ask next? "
                         "Focus on gathering more details about the symptoms mentioned. "
                         "If you have enough information, provide a brief assessment and recommendation.\n\n"
                         "Assistant: ")
        
        # Combine all parts into a single prompt
        prompt = "\n".join(prompt_parts)
        
        # Generate response using Gemini
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 150,
                "top_p": 0.9,
                "top_k": 40,
            }
        )
        
        if response and hasattr(response, 'text') and response.text.strip():
            # Clean up the response to remove any role prefixes
            response_text = response.text.strip()
            if response_text.startswith("Assistant:"):
                response_text = response_text[10:].strip()
            return response_text
        
        return "I apologize, but I'm having trouble generating a response. Could you please rephrase your last statement?"
        
    except Exception as e:
        print(f"Error in get_next_question: {e}")
        # Fallback to a simple question if there's an error
        return "Could you please tell me more about your symptoms?"

def generate_final_recommendation():
    """Generate a concise medical summary with symptoms, suggestions, and medication advice"""
    try:
        # Extract key symptoms and information
        symptoms = set()
        age = None
        
        for msg in st.session_state.conversation:
            if msg["role"] == "user":
                content = msg['content'].lower()
                # Extract symptoms
                if any(symptom in content for symptom in ['fever', 'headache', 'pain', 'nausea', 'vomit', 'cough', 'sore throat', 'fatigue']):
                    symptoms.add(next((s for s in ['fever', 'headache', 'pain', 'nausea', 'vomit', 'cough', 'sore throat', 'fatigue'] if s in content), 'general discomfort'))
                # Extract age if mentioned
                if 'age' in content:
                    try:
                        age = int(''.join(filter(str.isdigit, content)))
                    except:
                        pass
        
        # Prepare the prompt for a concise response
        prompt = f"""Provide a 2-3 line medical summary including:
        1. Key symptoms: {', '.join(symptoms) if symptoms else 'Not specified'}
        2. Suggested actions (self-care or see a doctor)
        3. Any OTC medication suggestions if applicable
        
        Keep it very brief and to the point. If age {age} is relevant, consider it in your response."""
        
        # Generate response using the LLM
        response = model.generate_content(prompt)
        
        # Format the response to be more concise if needed
        if hasattr(response, 'text'):
            text = response.text
        elif hasattr(response, 'result'):
            text = response.result
        else:
            text = "For persistent or severe symptoms, please consult a healthcare professional."
        
        # Ensure the response is brief (2-3 lines)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines[:3])
        
    except Exception as e:
        print(f"Error generating recommendation: {e}")
        return "Please consult a healthcare professional for persistent symptoms."
        # Return a fallback question if there's an error
        question_index = hash(str(conversation_history) + str(e)) % len(fallback_questions)
        return fallback_questions[question_index]

def save_conversation(conversation, filename=None):
    """Save conversation to a JSON file"""
    if filename is None:
        filename = f"conversation_{get_timestamp()}.json"
    
    filepath = DATA_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(conversation, f, indent=2)
    return filepath

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="MediBot",
        page_icon="👨‍⚕️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 MediBot")
    
    # Initialize TTS engine once at the start
    if VOICE_AVAILABLE and not _tts_initialized:
        init_tts_engine()
    
    # Initialize session state
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
        st.session_state.symptom_summary = {}
        st.session_state.asked_questions = set()
        st.session_state.conversation_history = []
        st.session_state.user_input = ""
        st.session_state.last_spoken = ""
        st.session_state.needs_to_speak = None
        st.session_state.listening = False
        st.session_state.first_run = True
    
    # Add initial greeting and first question if this is the first run
    if not st.session_state.conversation:
        initial_greeting = "Welcome! I'm here to help assess your symptoms. Please answer the following questions."
        first_question = "What is your age?"
        
        # Add greeting to conversation
        st.session_state.conversation.append({"role": "assistant", "content": initial_greeting})
        
        # Add first question to conversation
        st.session_state.conversation.append({"role": "assistant", "content": first_question})
        
        # Set up speech state
        st.session_state.asked_questions.add('age')
        
        # Initialize TTS engine if voice is available
        if VOICE_AVAILABLE:
            init_tts_engine()
            # Speak both the welcome message and first question
            speak(initial_greeting)
            speak(first_question)
        
        # Set the last spoken to the first question to prevent duplicates
        st.session_state.last_spoken = first_question
    
    # Debug information
    if st.sidebar.checkbox("Show debug info"):
        st.sidebar.write("## Debug Info")
        st.sidebar.json({"session_state_keys": list(st.session_state.keys())})
        
        if 'conversation' in st.session_state:
            st.sidebar.write("### Conversation History")
            st.sidebar.json([{"role": m["role"], "content": m["content"][:50] + "..." if len(m["content"]) > 50 else m["content"]} for m in st.session_state.conversation])
    
    # Display conversation
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Track the last assistant message that was spoken to prevent duplicates
    last_assistant_message = ""
    if st.session_state.conversation and st.session_state.conversation[-1]["role"] == "assistant":
        last_assistant_message = st.session_state.conversation[-1]["content"]
    
    # Check if we need to speak the last assistant message
    if (VOICE_AVAILABLE and 
        'last_spoken' in st.session_state and 
        st.session_state.last_spoken != last_assistant_message and 
        last_assistant_message):
        st.session_state.last_spoken = last_assistant_message
        speak(last_assistant_message)
    
    # Check if we need to ask the next question
    if st.session_state.conversation and st.session_state.conversation[-1]["role"] == "user":
        user_input = st.session_state.conversation[-1]["content"]
        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation])
        
        # Check if user is asking for a summary
        if any(phrase in user_input.lower() for phrase in ["summary", "what's wrong", "what do i have", "conclude"]):
            ai_response = generate_final_recommendation()
        else:
            # Get next question or response
            ai_response = get_next_question(conversation_history, user_input)
        
        if ai_response:
            # Add assistant's response to conversation
            st.session_state.conversation.append({"role": "assistant", "content": ai_response})
            st.session_state.last_spoken = ai_response
            
            # Speak the response if voice is available
            if VOICE_AVAILABLE and ai_response.strip():
                speak(ai_response)
            
            st.rerun()  # Rerun to show the new message
    
    # Create a container for the input area at the bottom
    with st.container():
        # Voice input button (only show if voice is available)
        if VOICE_AVAILABLE and st.button("🎤 Speak", key="speak_button"):
            user_input = listen()
            if user_input:
                st.session_state.conversation.append({"role": "user", "content": user_input})
                st.rerun()
        
        # Single text input form
        with st.form(key="user_input_form", clear_on_submit=True):
            # Text input
            text_input = st.text_input(
                "Type your response here:",
                key="text_input",
                value="",
                label_visibility="collapsed"
            )
            
            # Create two columns for the buttons
            col1, col2 = st.columns([1, 1])
            
            # Submit button for sending the message
            with col1:
                submitted = st.form_submit_button("Send")
            
            # Clear button
            with col2:
                clear_clicked = st.form_submit_button("Clear")
            
            # Handle form submission
            if submitted or clear_clicked:
                if clear_clicked:
                    st.session_state.user_input = ""
                    st.rerun()
                elif submitted and text_input.strip():
                    user_input = text_input.strip()
                    st.session_state.conversation.append({"role": "user", "content": user_input})
                    st.rerun()
                    
        # Speak any pending messages
        if VOICE_AVAILABLE and st.session_state.get('needs_to_speak'):
            text_to_speak = st.session_state.needs_to_speak
            st.session_state.needs_to_speak = None  # Clear the flag
            
            def _speak_safely(text):
                try:
                    speak(text)
                except Exception as e:
                    print(f"Error in speech thread: {e}")
            
            import threading
            try:
                # Stop any existing speech thread
                for thread in threading.enumerate():
                    if thread.name.startswith('SpeechThread'):
                        thread.join(timeout=0.1)
                
                # Start new speech thread
                speech_thread = threading.Thread(
                    target=_speak_safely,
                    args=(text_to_speak,),
                    daemon=True,
                    name=f"SpeechThread_pending"
                )
                speech_thread.start()
            except Exception as e:
                print(f"Error starting speech thread: {e}")
    
    # Check if we need to ask the next question
    if st.session_state.conversation and st.session_state.conversation[-1]["role"] == "user":
        user_input = st.session_state.conversation[-1]["content"]
        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation])
        
        # Check if user is asking for a summary
        if any(phrase in user_input.lower() for phrase in ["summary", "what's wrong", "what do i have", "conclude"]):
            response = generate_final_recommendation()
        else:
            # Get next question or response
            response = get_next_question(conversation_history, user_input)
        
        if response:
            # Add assistant's response to conversation
            st.session_state.conversation.append({"role": "assistant", "content": response})
            st.session_state.last_spoken = response
            
            # Start speech in a separate thread if voice is available
            if VOICE_AVAILABLE and response.strip():
                def _speak_safely(text):
                    try:
                        speak(text)
                    except Exception as e:
                        print(f"Error in speech thread: {e}")
                
                import threading
                try:
                    # Stop any existing speech thread
                    for thread in threading.enumerate():
                        if thread.name.startswith('SpeechThread'):
                            thread.join(timeout=0.1)
                    
                    # Start new speech thread
                    speech_thread = threading.Thread(
                        target=_speak_safely,
                        args=(response,),
                        daemon=True,
                        name=f"SpeechThread_{len(st.session_state.conversation)}"
                    )
                    speech_thread.start()
                    
                    # Small delay to prevent UI freezing
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error starting speech thread: {e}")
            
            # Rerun to show the new message
            st.rerun()
    
    # Save conversation when done
    if st.button("End Session"):
        try:
            # Save the current conversation
            filename = f"conversation_{get_timestamp()}.json"
            save_path = save_conversation(st.session_state.conversation, filename)
            st.success(f"Conversation saved to {save_path}")
            
            # Clear all session state
            st.session_state.clear()
            
            # Reinitialize with welcome message and first question
            initial_greeting = "Welcome! I'm here to help assess your symptoms. Please answer the following questions."
            first_question = "What is your age?"
            
            st.session_state.conversation = [
                {"role": "assistant", "content": initial_greeting},
                {"role": "assistant", "content": first_question}
            ]
            st.session_state.asked_questions = {'age'}
            st.session_state.last_spoken = first_question
            
            # Speak the messages
            if VOICE_AVAILABLE:
                speak(initial_greeting)
                speak(first_question)
            
            st.rerun()
        except Exception as e:
            st.error(f"Error saving conversation: {e}")

if __name__ == "__main__":
    main()

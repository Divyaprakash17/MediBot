# 🏥 AI-Powered Medical Chatbot with Voice Interface

A comprehensive medical chatbot that provides symptom assessment using Google's Gemini AI, featuring voice interaction, dynamic questioning, and conversation history.

## ✨ Features

### Core Functionality
- **Dynamic Symptom Assessment**: AI-powered conversation flow that adapts to user responses
- **Multi-modal Interaction**: Supports both text and voice input/output
- **Conversation History**: Automatic logging of all interactions
- **Responsive Web Interface**: Built with Streamlit for seamless user experience

### Advanced Capabilities
- **Voice Recognition**: Real-time speech-to-text conversion
- **Text-to-Speech**: Natural-sounding voice responses
- **Contextual Understanding**: Maintains conversation context for relevant follow-ups
- **Symptom Analysis**: Provides preliminary health assessments

### Security & Privacy
- Local processing of voice data
- Optional cloud storage for conversation history
- Secure API key management

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Microphone (for voice input)
- Speakers/headphones (for voice output)
- [Google Gemini API Key](https://ai.google.dev/)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Divyaprakash17/MediBot.git
   cd medical-chatbot
   ```

2. **Set up virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   Create a `.env` file in the project root:
   ```env
   # Required
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Optional (for cloud storage)
   FIREBASE_CREDENTIALS=path/to/your/firebase-credentials.json
   ```

## 🖥️ Usage

### Local Development
1. Start the application:
   ```bash
   streamlit run medical_chatbot.py
   ```
2. Open your browser to `http://localhost:8501`

### Voice Commands
- Click the microphone button to start voice input
- Speak clearly when the microphone is active
- Click again to stop recording

### Text Input
- Type your response in the text box
- Press Enter or click 'Send' to submit

## 🛠️ Project Structure

```
medical_chatbot/
├── chat_history/           # Stores conversation history in JSON format
│   ├── conversation_*.json  # Individual chat session logs
├── medical_chatbot.py      # Main application file
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## 🚀 Running the Application

1. **Start the application**:
   ```bash
   python medical_chatbot.py
   ```

2. **Access the interface** in your web browser at `http://localhost:8501`

3. **Using the chatbot**:
   - Type your responses in the text input
   - Or use the microphone button for voice input
   - Press Enter or click 'Send' to submit

## 📝 Features in Detail

### Dynamic Questioning
- Adapts questions based on previous answers
- Follows medical assessment protocols
- Handles various response formats (text/voice)

### Voice Processing
- Real-time speech recognition
- Natural language understanding
- Clear text-to-speech responses

### Data Management
- Local JSON storage by default
- Optional cloud storage with Firebase
- Secure handling of sensitive information



## 🙏 Acknowledgments

- Google Gemini for the AI capabilities
- Streamlit for the web interface
- speech_recognition for speech recognition
- pyttsx3 for text to speech

---

<div align="center">
Made with ❤️ for better healthcare accessibility
</div>

1. Run the Streamlit application:
   ```bash
   streamlit run medical_chatbot.py
   ```

2. The application will open in your default web browser.

3. Use the "Speak" button to provide voice input or type your responses in the text field.

4. Click "End Session" when finished to save the conversation.

## How It Works

1. The chatbot starts by greeting the user and asking about their symptoms.
2. Based on the user's responses, the Gemini AI generates relevant follow-up questions.
3. The conversation continues until the user ends the session.
4. The entire conversation is saved to a JSON file in the `chat_history` directory.

## File Structure

- `medical_chatbot.py`: Main application file
- `requirements.txt`: Python dependencies
- `README.md`: This file
- `chat_history/`: Directory where conversation logs are stored

## Important Notes

- This is not a substitute for professional medical advice.
- Always consult a healthcare professional for medical concerns.
- Conversations are saved locally for reference but are not shared with any third parties.

## Troubleshooting

- If you encounter issues with voice recognition, ensure your microphone is properly connected and permissions are granted.
- Make sure you have set the `GEMINI_API_KEY` in your `.env` file.



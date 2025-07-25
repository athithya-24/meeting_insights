# Meeting Insights Pro

An advanced web application that leverages AI to transcribe, analyze, and summarize meeting audio with comprehensive insights.

##  Features

- **Automated Transcription**: High-accuracy speech-to-text using OpenAI Whisper with timestamps
- **AI-Powered Summaries**: Comprehensive meeting summaries via Google Gemini
- **Action Item Extraction**: Automatic identification of key decisions and follow-ups
- **Sentiment Analysis**: Real-time sentiment tracking with visual charts
- **Word Analytics**: Frequency analysis and word cloud generation
- **Multiple Export Formats**: PDF, DOCX, TXT, and SRT subtitle support
- **Meeting History**: Automatic archiving and management
- **Modern UI**: Clean, responsive single-page interface

##  Prerequisites

- Python 3.8+
- FFmpeg (must be in system PATH)
- Google Gemini API key

##  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/meeting-insights-pro.git
   cd meeting-insights-pro
   ```

2. **Install dependencies**
   ```bash
   pip install flask whisper google-generativeai python-dotenv reportlab matplotlib wordcloud numpy python-docx
   ```

3. **Environment setup**
   
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your-gemini-api-key
   ```

4. **Run the application**
   ```bash
   python app1.py
   ```

5. **Access the app** at `http://localhost:5000`

##  Usage

1. Click **'Record'** to start capturing meeting audio
2. Wait for automatic transcription and AI analysis
3. Review transcript, summary, and insights
4. Export results in your preferred format

##  API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page interface |
| `/process` | POST | Upload audio for analysis |
| `/history` | GET | List saved meetings |
| `/history/<filename>` | GET | Get specific meeting data |
| `/export/<format>/<type>` | GET | Export transcripts/summaries |
| `/health` | GET | Server health check |

##  Project Structure

```
meeting-insights-pro/
├── app1.py                 # Main Flask application
├── meeting_history/        # Saved meeting data
├── .env                   # Environment variables
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

##  Security Features

- File validation (audio files only, 50MB limit)
- Local audio processing for privacy
- Secure API key management
- Input validation and error handling

##  Use Cases

- Project meetings and brainstorming sessions
- Interview recordings and research
- Academic seminars and focus groups
- Corporate and remote team discussions

##  Acknowledgments

Built with Flask, OpenAI Whisper, and Google Gemini for intelligent meeting analysis and productivity enhancement.

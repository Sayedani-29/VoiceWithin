# 🧠 VoiceWithin

> **A mental health companion web app for students** — journaling, AI chat, risk assessment, and mood tracking, all in one place.

VoiceWithin is a Flask-based web application designed to support student mental health. It combines a mood journal, an AI-powered chatbot, a mental health risk assessment survey, and push notifications into a single, empathetic platform.

---

## ✨ Features

### 🤖 AI Chatbot
- Powered by **Groq's LLaMA 3.3 70B** model
- Responds in short, warm, Gen-Z-friendly tones — like a supportive friend at 2AM
- **Crisis detection**: instantly surfaces the AASRA helpline (🇮🇳 9820466627) if distress signals are detected
- **Short-term memory**: retains the last 10 messages per chat for context
- Auto-names chats based on detected emotions (e.g. "Stressed Talk", "Happy Chat")
- NLTK VADER sentiment analysis runs on every user message

### 📓 Mood Journal
- Write daily entries with a title, content, and mood tag
- Attach images (stored in MongoDB GridFS)
- View, edit, and delete past entries
- Mood statistics with daily and monthly trend summaries

### 📊 Mental Health Risk Assessment
- Multi-question survey covering depression, anxiety, panic attacks, suicidal ideation, self-harm, bullying, substance use, and more
- ML model (scikit-learn) predicts risk level: **Low / Medium / High**
- Tailored suggestions based on prediction
- History tracked per user for the improvement dashboard

### 📈 Improvement Dashboard
- Combines QA risk history and mood journal data
- Shows progress over time with visual charts

### 🔔 Push Notifications
- Web Push via VAPID keys
- Motivational messages sent at **10 AM** and **7 PM** daily using APScheduler

---

## 🗂️ Project Structure

```
VoiceWithin/
├── hie.py                    # Main Flask app (routes, chatbot, ML, notifications)
├── fixed_chat_route.py       # Isolated chat route (dev iteration artifact)
├── requirements.txt          # Python dependencies
├── Procfile                  # Render deployment config
├── suicide_model(3).pkl      # Trained ML model (scikit-learn)
├── label_encoders.pkl(3)     # Label encoders for ML features
├── .gitignore
└── templates/
    ├── intro.html            # Landing page
    ├── index.html            # Signup page
    ├── login.html            # Login page
    ├── main_menu.html        # Dashboard
    ├── survey.html           # Mental health QA form
    ├── QA.html               # QA interface
    ├── result.html           # Prediction result
    ├── journal.html          # New journal entry
    ├── hist.html             # Journal history
    ├── chatbot.html          # AI chat interface
    └── improve.html          # Progress dashboard
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Database | MongoDB Atlas (PyMongo, GridFS) |
| AI / LLM | Groq API — LLaMA 3.3 70B |
| ML Model | scikit-learn (pickle) |
| Sentiment | NLTK VADER |
| Push Notifications | pywebpush + APScheduler |
| Auth | Flask sessions |
| Deployment | Render (free tier) |
| Frontend | HTML, CSS, JavaScript |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- MongoDB Atlas account
- Groq API key ([get one here](https://console.groq.com))
- VAPID keys for Web Push (generate with `py-vapid` or `web-push-codelab`)

### 1. Clone the repo

```bash
git clone https://github.com/Sayedani-29/VoiceWithin.git
cd VoiceWithin
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

You may also need to download the NLTK lexicon:

```python
import nltk
nltk.download("vader_lexicon")
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
MONGO_URI=your_mongodb_connection_string_here
SECRET_KEY=your_flask_secret_key_here
```

> ⚠️ Never commit your `.env` file. It's already included in `.gitignore`.

### 4. Run the app

```bash
python hie.py
```

Visit `http://localhost:5000` in your browser.

---

## ☁️ Deployment (Render)

This project is configured for deployment on [Render](https://render.com).

The `Procfile` tells Render how to start the app:

```
web: python hie.py
```

**Steps:**

1. Push your code to GitHub.
2. Create a new **Web Service** on Render and connect your repository.
3. Add the following environment variables in Render's dashboard:
   - `GROQ_API_KEY`
   - `MONGO_URI`
   - `SECRET_KEY`
4. Deploy.

> 💡 The app includes a keep-alive thread that pings itself every 14 minutes to prevent Render's free tier from sleeping.

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | API key for Groq LLM access |
| `MONGO_URI` | MongoDB Atlas connection string |
| `SECRET_KEY` | Flask session secret key |

---

## 🧩 API Routes

| Method | Route | Description |
|---|---|---|
| GET | `/` | Intro / landing page |
| GET/POST | `/index` | Signup |
| GET/POST | `/login` | Login |
| GET | `/logout` | Logout |
| GET | `/main_menu` | Dashboard (auth required) |
| GET | `/survey` | Mental health survey |
| POST | `/predict` | Run ML prediction |
| GET | `/result` | Show prediction result |
| GET | `/journal` | New journal entry form |
| POST | `/add` | Save journal entry |
| GET | `/hist` | Journal history |
| POST | `/edit/<id>` | Edit a journal entry |
| POST | `/delete/<id>` | Delete a journal entry |
| GET | `/mood_stats` | Mood statistics (JSON) |
| GET | `/improve` | Improvement dashboard |
| GET | `/chatbot` | Chatbot UI |
| POST | `/chat` | Send a message to the bot |
| POST | `/save_subscription` | Register push notification subscription |
| GET | `/health` | Health check endpoint |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get involved:

1. **Fork** this repository
2. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and commit with a clear message:
   ```bash
   git commit -m "feat: add your feature description"
   ```
4. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request** against the `master` branch

### Ideas for contribution
- Add password hashing (bcrypt)
- Add email verification on signup
- Build a more detailed progress dashboard
- Add multilingual support
- Improve mobile responsiveness

---

## 🐛 Known Issues & Security Notes

> These are known limitations to be aware of before deploying to production:

- **Passwords are stored in plaintext** — use `bcrypt` or `werkzeug.security` to hash passwords before storing.
- **VAPID keys are hardcoded** in `hie.py` — move them to environment variables.
- **Sessions are not encrypted beyond Flask's default** — set a strong, random `SECRET_KEY`.
- `fixed_chat_route.py` is a development artifact and is not used in production.

---

## 💛 Acknowledgements

- [Groq](https://groq.com) — blazing fast LLM inference
- [NLTK](https://www.nltk.org) — sentiment analysis
- [MongoDB Atlas](https://www.mongodb.com/atlas) — cloud database
- [AASRA](http://www.aasra.info) — crisis helpline integrated for Indian users (24/7: 9820466627)
- Everyone who contributed to making mental health tools more accessible 💛

---
## 📄 License

***© 2024 Sayedani-29. All rights reserved. This project and its contents may not be reproduced, distributed, or used without explicit permission from the author.***
---
> *"Your story matters. You matter."* 🌻

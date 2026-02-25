# ==========================================================
# Combined Flask App: QA + Mood Journal + Chatbot + Notifications
# ==========================================================

from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, session
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import gridfs
import io
import pickle
import numpy as np
import random, os, json
from dotenv import load_dotenv
from collections import defaultdict, Counter
from pywebpush import webpush
from apscheduler.schedulers.background import BackgroundScheduler
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
from groq import Groq


# ==========================================================
# Config & Setup
# ==========================================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("=" * 50)
print("🔑 GROQ API KEY CHECK")
print(f"Key exists: {GROQ_API_KEY is not None}")
print(f"Key starts with: {GROQ_API_KEY[:10] if GROQ_API_KEY else 'MISSING'}")
print("=" * 50)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fallback-dev-key") # change in production

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["SSP"]

# Collections
entries_col = db["Mood Journal"]
users_col = db["users"]
qa_col = db["QA"]
chats = db["Chatbot"]
fs = gridfs.GridFS(db, collection="images")

# VAPID keys for notifications
VAPID_PUBLIC_KEY = "BK5IfpSj2hRtPdW13Q_66kLtypJSDpValC5LoED7ylls4ECXz9reQm9CIyp35uc2kCbnZImnQtv0eU9oYCMDll8"
VAPID_PRIVATE_KEY = "g9NKsBaz2wbb313JhSShqQg5sbZMmbMyd_K2GEt483c"
VAPID_CLAIMS = {"sub": "mailto:youremail@example.com"}

# Motivational messages
MESSAGES = [
    "🌞 Start your day strong! Every small step counts.",
    "💬 How are you feeling today? Take a moment to reflect.",
    "🌻 Remember, it's okay to slow down and breathe.",
    "🌙 End your day with gratitude — you did your best.",
    "💡 Keep going, your story matters!",
    "🌼 You're stronger than you think. Believe in yourself.",
    "🌈 Even tough days teach valuable lessons.",
]

# Load ML model & encoders
with open("suicide_model(3).pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoders.pkl(3)", "rb") as f:
    label_encoders = pickle.load(f)

# ==========================================================
# Helper Functions
# ==========================================================
def doc_to_public(doc):
    return {
        "_id": str(doc.get("_id")),
        "email": doc.get("email"),
        "title": doc.get("title", ""),
        "content": doc.get("content", ""),
        "mood": doc.get("mood", "neutral"),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
        "images": [f"/image/{iid}" for iid in doc.get("image_ids", []) if iid]
    }

def require_login_redirect():
    if "email" not in session:
        return redirect(url_for("login"))
    return None

# SMART CHAT NAMING
def generate_smart_chat_name(user_message):
    emotions = {
        "stress": ["stress", "stressed", "overwhelmed", "pressure", "anxious", "worry"],
        "sad": ["sad", "down", "depressed", "lonely", "empty", "hurt"],
        "happy": ["happy", "great", "excited", "good", "amazing"],
        "angry": ["angry", "mad", "frustrated", "upset", "rage"]
    }
    
    user_msg = user_message.lower()
    for emotion, keywords in emotions.items():
        if any(word in user_msg for word in keywords):
            return f"{emotion.title()} Talk"
    
    starters = ["Today", "Feeling", "Chat", "Thoughts"]
    return f"{random.choice(starters)} {datetime.now().strftime('%a %H:%M')}"

# CRISIS DETECTION (KEEP YOUR SAFETY NET)
def crisis_check(text):
    crisis_words = ["suicide", "kill", "self harm", "end my life", "die", "dead"]
    if any(w in text.lower() for w in crisis_words):
        return "I'm really sorry you're feeling this much pain. You deserve support and care.\n\n🇮🇳 AASRA Helpline (24/7): 9820466726"
    return None

# ==========================================================
# GROQ CHATBOT CONFIG (CLEAN & REUSABLE)
# ==========================================================
groq_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
You are VoiceWithin, a supportive emotional bestie.
You respond in 1–3 short sentences ONLY.
You listen more than you speak.
You do NOT explain, lecture, or give long advice.
You validate feelings and sound like a cozy gen-z friend texting at 2AM 🤗💛
If pain feels heavy, gently encourage support — never overwhelm.
"""

def build_chat_context(chat_name, user_message):
    """
    Fetch last few messages from MongoDB
    and prepare context for Groq
    """
    context = []

    chat_doc = chats.find_one({"chat_name": chat_name})
    if chat_doc and "messages" in chat_doc:
        for msg in chat_doc["messages"][-10:]:  # short-term memory
            role = "user" if msg["sender"] == "user" else "assistant"
            context.append({
                "role": role,
                "content": msg["text"]
            })

    context.append({"role": "user", "content": user_message})
    return context


def generate_groq_reply(context_messages):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *context_messages
        ],
        temperature=0.5,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()

# ==========================================================
# Routes: Groq Test
# ==========================================================
@app.route("/test_groq")
def test_groq():
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello World' and nothing else."}
            ],
            max_tokens=50
        )
        
        reply = response.choices[0].message.content
        return jsonify({
            "status": "✅ SUCCESS",
            "reply": reply,
            "model": "llama-3.3-70b-versatile"
        })
    
    except Exception as e:
        return jsonify({
            "status": "❌ FAILED",
            "error": str(e)
        }), 500

# ==========================================================
# Routes: Checking MongoDB 
# ==========================================================    
@app.route("/test_mongo")
def test_mongo():
    try:
        # Test connection
        db.command('ping')
        
        # Test chatbot collection
        test_doc = {
            "chat_name": "test_chat",
            "messages": [
                {"sender": "user", "text": "test", "timestamp": datetime.now().isoformat()}
            ]
        }
        result = chats.insert_one(test_doc)
        
        # Clean up
        chats.delete_one({"_id": result.inserted_id})
        
        return jsonify({
            "status": "✅ MongoDB Connected",
            "database": db.name,
            "collection": "Chatbot"
        })
    
    except Exception as e:
        return jsonify({
            "status": "❌ MongoDB Failed",
            "error": str(e)
        }), 500

# ==========================================================
# Routes: Auth & Navigation
# ==========================================================
@app.route("/")
def intro():
    return render_template("intro.html")

@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if password != confirm_password:
            return jsonify({"status": "error", "message": "Passwords do not match"})
        if not name or not email or not password:
            return jsonify({"status": "error", "message": "All fields are required"})
        if users_col.find_one({"email": email}):
            return jsonify({"status": "error", "message": "Email already registered. Please login."})

        users_col.insert_one({"name": name, "email": email, "password": password})
        print("✅ Signup successful for:", email)
        return jsonify({"status": "success", "email": email})
    
    return render_template("index.html", public_key=VAPID_PUBLIC_KEY)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = users_col.find_one({"email": email, "password": password})
        if not user:
            return "Invalid credentials", 401
        session["email"] = user["email"]
        session["name"] = user.get("name", "")
        return redirect(url_for("main_menu"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("intro"))

@app.route("/main_menu")
def main_menu():
    r = require_login_redirect()
    if r: return r
    return render_template("main_menu.html", name=session.get("name"))
@app.route("/survey")
def survey():
    r = require_login_redirect()
    if r: return r
    return render_template("survey.html")

@app.route("/QA")
def QA():
    r = require_login_redirect()
    if r: return r
    return render_template("QA.html")

@app.route("/predict", methods=["POST"])
def predict():
    r = require_login_redirect()
    if r: return r

    data = request.get_json(force=True)
    email = session["email"]

    feature_order = [
        "Choose your gender", "Age", "What is your course?", "Your current year",
        "Do you have Depression?", "Do you have Anxiety?", "Do you have Panic attack?",
        "Did you seek any specialist for a treatment?",
        "Have you ever experienced thoughts of ending your life (suicidal ideation)?",
        "Have you ever attempted to end your life in the past?",
        "Have you ever intentionally harmed yourself (e.g., cutting, burning, etc.)?",
        "Do you currently feel hopeless about your future?",
        "Have you ever experienced bullying (in person or online) that negatively affected your mental health?",
        "Are you currently experiencing serious family-related issues (e.g., conflict, abuse, divorce, financial stress)?",
        "Have you used substances (e.g., alcohol, tobacco, recreational drugs) to cope with stress or emotions?"
    ]

    input_data = []
    for feature in feature_order:
        val = data.get(feature, "")
        if feature in label_encoders:
            le = label_encoders[feature]
            try:
                transformed = le.transform([val])[0]
            except Exception:
                transformed = le.transform([le.classes_[0]])[0]
            input_data.append(transformed)
        else:
            try:
                input_data.append(float(val))
            except Exception:
                input_data.append(0.0)

    arr = np.array(input_data).reshape(1, -1)
    pred = model.predict(arr)[0]

    if "Suicide Rate" in label_encoders:
        target_le = label_encoders["Suicide Rate"]
        try:
            pred_text = target_le.inverse_transform([pred])[0]
        except Exception:
            pred_text = str(pred)
    else:
        pred_text = str(pred)

    suggestions = {
       "Low": "You’re doing awesome! 🚀 Keep shining, stay hydrated, and don’t forget to laugh 😄.",
            "Medium": "It seems like you’re going through some ups and downs 🌤️. Talking it out with a counselor or friend could help.",
            "High": "Things might feel heavy 💔. Please don’t face it alone — talking to a trusted person or professional can make a huge difference 🌱."
        }

    suggestion = suggestions.get(pred_text, "Please seek help if you feel unwell.")

    qa_doc = {
        "email": email,
        "timestamp": datetime.utcnow(),
        "answers": data,
        "risk_level": pred_text,
        "suggestion": suggestion
    }
    qa_col.insert_one(qa_doc)

    # Support older pattern where QA stored as a single doc with qa_history array:
    qa_col.update_one(
        {"email": email, "qa_history": {"$exists": True}},
        {"$push": {"qa_history": {"timestamp": datetime.utcnow(), "risk_level": pred_text, "answers": data, "suggestion": suggestion}}},
        upsert=False
    )

    session["prediction"] = pred_text
    session["suggestion"] = suggestion

    return jsonify({"redirect": url_for("result"), "prediction": pred_text, "suggestion": suggestion})

@app.route("/result")
def result():
    r = require_login_redirect()
    if r: return r
    prediction = session.get("prediction", "Not Available")
    suggestion = session.get("suggestion", "No suggestion available.")
    return render_template("result.html", prediction=prediction, suggestion=suggestion)

# ==========================================================
# Mood Journal Routes (unchanged)
# ==========================================================
@app.route("/journal")
def journal():
    r = require_login_redirect()
    if r: return r
    return render_template("journal.html")

@app.route("/add", methods=["POST"])
def add_entry():
    r = require_login_redirect()
    if r: return r

    title = request.form.get("title", "").strip() or "Untitled"
    content = request.form.get("content", "").strip()
    mood = request.form.get("mood", "neutral")
    created_at = datetime.utcnow()
    email = session.get("email")

    image_ids = []
    for file in request.files.getlist("image"):
        if file and file.filename:
            fid = fs.put(file.read(), filename=file.filename, content_type=file.mimetype)
            image_ids.append(str(fid))

    entries_col.insert_one({
        "email": email,
        "title": title,
        "content": content,
        "mood": mood,
        "image_ids": image_ids,
        "created_at": created_at
    })
    return redirect(url_for("history"))

@app.route("/entries")
def get_entries():
    if "email" not in session:
        return jsonify([])
    docs = entries_col.find({"email": session["email"]}).sort("created_at", -1)
    return jsonify([doc_to_public(d) for d in docs])

@app.route("/hist")
def history():
    r = require_login_redirect()
    if r: return r
    docs = entries_col.find({"email": session["email"]}).sort("created_at", -1)
    entries = [doc_to_public(d) for d in docs]
    return render_template("hist.html", entries=entries)

@app.route("/edit/<id>", methods=["POST"])
def edit_entry(id):
    r = require_login_redirect()
    if r: return r
    oid = ObjectId(id)
    doc = entries_col.find_one({"_id": oid, "email": session["email"]})
    if not doc:
        return "Not found", 404

    title = request.form.get("title", "").strip() or "Untitled"
    content = request.form.get("content", "").strip()
    mood = request.form.get("mood", "neutral")

    files = request.files.getlist("image")
    if files and any(f.filename for f in files):
        for iid in doc.get("image_ids", []):
            try:
                fs.delete(ObjectId(iid))
            except:
                pass
        new_ids = []
        for f in files:
            fid = fs.put(f.read(), filename=f.filename, content_type=f.mimetype)
            new_ids.append(str(fid))
        entries_col.update_one({"_id": oid}, {"$set": {"title": title, "content": content, "mood": mood, "image_ids": new_ids}})
    else:
        entries_col.update_one({"_id": oid}, {"$set": {"title": title, "content": content, "mood": mood}})

    return redirect(url_for("history"))

@app.route("/delete/<id>", methods=["GET", "POST"])
def delete_entry(id):
    print("🧾 DELETE route triggered with ID:", id)
    try:
        oid = ObjectId(str(id))
    except Exception as e:
        print("❌ Invalid ObjectId:", e)
        return jsonify({"error": "Invalid ID"}), 400

    doc = entries_col.find_one({"_id": oid})
    if not doc:
        print("❌ No document found for this ID")
        return jsonify({"error": "Not found"}), 404

    for iid in doc.get("image_ids", []):
        try:
            fs.delete(ObjectId(iid))
        except Exception as e:
            print("⚠️ Error deleting image:", e)

    result = entries_col.delete_one({"_id": oid})
    print("🗑️ Deleted count:", result.deleted_count)

    return jsonify({"success": True})

@app.route("/image/<image_id>")
def serve_image(image_id):
    try:
        grid_out = fs.get(ObjectId(image_id))
        return send_file(io.BytesIO(grid_out.read()), mimetype=grid_out.content_type)
    except:
        return "Image not found", 404

@app.route("/mood_stats")
def mood_stats():
    if "email" not in session:
        return jsonify({"daily_summary": [], "monthly_summary": []})
    docs = list(entries_col.find({"email": session["email"]}))
    daily = defaultdict(list)
    monthly = defaultdict(list)
    for d in docs:
        if not d.get("created_at"): continue
        day = d["created_at"].strftime("%Y-%m-%d")
        month = d["created_at"].strftime("%Y-%m")
        mood = d.get("mood", "neutral")
        daily[day].append(mood)
        monthly[month].append(mood)
    daily_summary = [{"date": k, "mood": Counter(v).most_common(1)[0][0]} for k, v in sorted(daily.items())]
    monthly_summary = [{"month": k, "mood": Counter(v).most_common(1)[0][0]} for k, v in sorted(monthly.items())]
    return jsonify({"daily_summary": daily_summary, "monthly_summary": monthly_summary})

@app.route("/improve")
def improvement():
    if "email" not in session:
        return redirect(url_for("login"))

    email = session["email"]

    try:
        qa_data = []
        qa_docs = list(qa_col.find({"email": email}))

        # ✅ Handle both styles of QA data
        for doc in qa_docs:
            if "qa_history" in doc:
                for entry in doc["qa_history"]:
                    ts = entry.get("timestamp")
                    risk = entry.get("risk_level", "Unknown")
                    if ts and risk:
                        qa_data.append({
                            "date": ts.strftime("%Y-%m-%d"),
                            "risk": risk
                        })
            else:
                ts = doc.get("timestamp")
                risk = doc.get("risk_level", "Unknown")
                if ts and risk:
                    qa_data.append({
                        "date": ts.strftime("%Y-%m-%d"),
                        "risk": risk
                    })

        qa_data.sort(key=lambda x: x["date"])

        # ✅ Get mood journal entries
        mood_entries = list(entries_col.find({"email": email}))
        mood_data = []
        for entry in mood_entries:
            created_at = entry.get("created_at")
            if created_at:
                mood_data.append({
                    "date": created_at.strftime("%Y-%m-%d"),
                    "mood": entry.get("mood", "")
                })

        last_risk = qa_data[-1]["risk"] if qa_data else "N/A"
        total_journal_entries = len(mood_data)

        return render_template(
            "improve.html",
            risk_data=qa_data,
            mood_data=mood_data,
            last_risk=last_risk,
            total_journal_entries=total_journal_entries
        )

    except Exception as e:
        print("Error in /improve:", e)
        return render_template(
            "improve.html",
            message="Error loading improvement data.",
            risk_data=[],
            mood_data=[],
            last_risk="N/A",
            total_journal_entries=0
        )

# ==========================================================
# AI CHATBOT (NEW GROQ-POWERED)
# ==========================================================
@app.route("/chatbot")
def chatbot_home():
    return render_template("chatbot.html")

# ==========================================================
# FIXED /chat ROUTE WITH ENHANCED STORAGE
# ==========================================================
@app.route("/chat", methods=["POST"])
def chat():
    print("\n" + "="*50)
    print("🔥 CHAT ROUTE HIT")
    print("="*50)
    
    try:
        # Get request data
        data = request.get_json()
        print(f"📩 Received data: {data}")
        
        user_message = data.get("message", "").strip()
        chat_name = data.get("chat_name", "default")
        
        print(f"💬 User message: '{user_message}'")
        print(f"📝 Chat name: '{chat_name}'")
        
        # Handle empty message
        if not user_message:
            print("⚠️ Empty message received")
            return jsonify({"reply": "Hey 🤗 what's on your mind?", "chat_name": chat_name})
        
        # 1️⃣ Crisis check (safety first)
        crisis = crisis_check(user_message)
        if crisis:
            print("🚨 Crisis message detected")
            return jsonify({
                "reply": crisis,
                "emotion": "crisis",
                "chat_name": chat_name
            })
        
        # 2️⃣ Auto-name chat if needed
        if chat_name == "default":
            chat_name = generate_smart_chat_name(user_message)
            print(f"✨ Generated chat name: {chat_name}")
        
        # 3️⃣ Build memory context from MongoDB
        context_messages = build_chat_context(chat_name, user_message)
        print(f"🧠 Context length: {len(context_messages)} messages")
        
        # 4️⃣ Get Groq response
        print("🤖 Calling Groq API...")
        
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *context_messages
                ],
                temperature=0.7,
                max_tokens=150,
                top_p=1,
                stream=False
            )
            bot_reply = response.choices[0].message.content.strip()
            print(f"✅ Bot reply: '{bot_reply}'")
            
        except Exception as groq_error:
            print(f"❌ GROQ API ERROR: {repr(groq_error)}")
            print(f"Error type: {type(groq_error).__name__}")
            print(f"Error details: {str(groq_error)}")
            
            # Return friendly error to user
            return jsonify({
                "reply": "Hey! I'm having trouble connecting right now 😅 Can you try again in a moment?",
                "chat_name": chat_name,
                "error": "groq_api_error"
            })
        
        # 5️⃣ Save messages to MongoDB with ENHANCED metadata
        try:
            # Analyze user's emotion
            sentiment = sia.polarity_scores(user_message)
            
            # Save USER message with metadata
            chats.update_one(
                {"chat_name": chat_name},
                {
                    "$push": {"messages": {
                        "sender": "user",
                        "text": user_message,
                        "timestamp": datetime.now().isoformat(),
                        
                        # 🆕 NEW METADATA
                        "emotion_scores": {
                            "positive": sentiment['pos'],
                            "negative": sentiment['neg'],
                            "neutral": sentiment['neu'],
                            "compound": sentiment['compound']
                        },
                        "word_count": len(user_message.split()),
                        "character_count": len(user_message),
                        "contains_crisis_words": bool(crisis_check(user_message)),
                        "message_id": str(ObjectId())
                    }},
                    
                    # 🆕 Also store chat-level metadata
                    "$set": {
                        "last_updated": datetime.now().isoformat(),
                        "email": session.get("email", "guest")
                    }
                },
                upsert=True
            )
            
            # Save BOT message with metadata
            chats.update_one(
                {"chat_name": chat_name},
                {"$push": {"messages": {
                    "sender": "bot",
                    "text": bot_reply,
                    "timestamp": datetime.now().isoformat(),
                    
                    # 🆕 NEW METADATA
                    "emotion": "empathetic",
                    "word_count": len(bot_reply.split()),
                    "message_id": str(ObjectId())
                }}},
                upsert=True
            )
            
            print("💾 Enhanced messages saved to MongoDB")
            
        except Exception as db_error:
            print(f"⚠️ MongoDB save error: {repr(db_error)}")
            # Continue anyway - user still gets response
        
        print("="*50)
        print("✅ CHAT COMPLETED SUCCESSFULLY")
        print("="*50 + "\n")
        
        return jsonify({
            "reply": bot_reply,
            "emotion": "empathetic",
            "chat_name": chat_name
        })
    
    except Exception as e:
        print("\n" + "="*50)
        print("❌ FULL CHAT ERROR")
        print("="*50)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*50 + "\n")
        
        return jsonify({
            "reply": "Oops! Something went wrong 😅 Please try again.",
            "error": str(e),
            "error_type": type(e).__name__
        }), 500



@app.route("/ping")
def ping():
    return "✅ Flask is running and ready!"

# Notifications
@app.route("/save_subscription", methods=["POST"])
def save_subscription():
    data = request.get_json()
    email = data.get("email")
    sub = data.get("subscription")
    if not email or not sub:
        return jsonify({"error": "Missing email or subscription"}), 400
    db.subscriptions.update_one(
        {"email": email},
        {"$set": {"subscription": sub}},
        upsert=True
    )
    print(f"📬 Saved push subscription for {email}")
    return jsonify({"status": "success"})

def send_push_notification():
    message = random.choice(MESSAGES)
    data = {"title": "VoiceWithin 💬", "body": message}
    for sub_doc in db.subscriptions.find():
        sub = sub_doc.get("subscription")
        if sub:
            try:
                webpush(
                    subscription_info=sub,
                    data=json.dumps(data),
                    vapid_private_key=VAPID_PRIVATE_KEY,
                    vapid_claims=VAPID_CLAIMS
                )
            except Exception as e:
                print("❌ Push failed:", e)
    print(f"✅ Notification sent: {message}")

@app.route("/test")
def test_push():
    send_push_notification()
    return "✅ Test notification sent"

# Start scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(send_push_notification, "cron", hour=10, minute=0)
scheduler.add_job(send_push_notification, "cron", hour=19, minute=0)
scheduler.start()

if __name__ == "__main__":
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port)

# ==========================================================
# FIXED CHATBOT ROUTE - Replace your /chat route with this
# ==========================================================

from flask import Flask, request, jsonify
from groq import Groq
import os
from datetime import datetime
import random
from dotenv import load_dotenv

# Initialize Groq client
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Updated system prompt
SYSTEM_PROMPT = """
You are VoiceWithin, a supportive emotional bestie.
Your replies are always short (1-3 sentences), warm, cozy, slightly silly, playful and genz.
Use emojis freely 🤗💛.
Never give long explanations, tables, or formal advice.
Always sound like a friend texting at 2 AM - comforting, fun, and human.
"""

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

def crisis_check(text):
    crisis_words = ["suicide", "kill", "self harm", "end my life", "die", "dead"]
    if any(w in text.lower() for w in crisis_words):
        return "I'm really sorry you're feeling this much pain. You deserve support and care.\n\n🇮🇳 AASRA Helpline (24/7): 9820466726"
    return None

def build_chat_context(chat_name, user_message, chats_collection):
    """Fetch last few messages from MongoDB"""
    context = []
    
    chat_doc = chats_collection.find_one({"chat_name": chat_name})
    if chat_doc and "messages" in chat_doc:
        for msg in chat_doc["messages"][-10:]:  # last 10 messages
            role = "user" if msg["sender"] == "user" else "assistant"
            context.append({
                "role": role,
                "content": msg["text"]
            })
    
    context.append({"role": "user", "content": user_message})
    return context

def generate_groq_reply(context_messages):
    """Call Groq API with CORRECT model name"""
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ FIXED: Using valid Groq model
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *context_messages
            ],
            temperature=0.7,
            max_tokens=150,  # ✅ FIXED: Increased from 40
            top_p=1,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ GROQ API ERROR: {str(e)}")
        return f"Oops, something went wrong 😅 Can you try again?"

# ==========================================================
# FIXED /chat ROUTE
# ==========================================================

"""
========================================
ORION - The Ultimate Chatbot
========================================
Multi-Model AI Chatbot (Streamlit Only)
----------------------------------------
Text answers: OpenAI, Gemini, Groq
Image generation: Stable Diffusion (Hugging Face)
Vision Q&A: OpenAI, Gemini
Audio to Text/Translate: OpenAI
----------------------------------------

"""

import os
import base64
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Streamlit required for this version

import streamlit as st 

# Load env vars from .env file

load_dotenv()

# Optional client imports (can be None in dev)

try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq as GroqClient
except ImportError:
    GroqClient = None

# -------------------------
# Config
# -------------------------

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
OPENAI_TEXT_MODEL = "gpt-4o-mini"
GEMINI_TEXT_MODEL = "gemini-1.5-flash"
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"

STABLE_DIFFUSION_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
HF_API_KEY = os.getenv("HF_API_KEY")

# -------------------------
# Response Wrapper
# -------------------------

@dataclass
class ProviderResponse:
    text: Optional[str] = None
    image_bytes: Optional[bytes] = None
    extra: Optional[Dict[str, Any]] = None

# -------------------------
# Providers (text API now accepts a `messages` list)
# -------------------------

class OpenAIProvider:
    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        if OpenAIClient is None:
            raise RuntimeError("openai package not installed")
        self.client = OpenAIClient(api_key=key)

    def text(self, messages: List[Dict[str, str]]) -> ProviderResponse:
        # Convert to OpenAI-style messages
        oa_msgs = []
        for m in messages:
            # messages stored with keys: role ('system'|'user'|'assistant') and content
            oa_msgs.append({"role": m["role"], "content": m["content"]})
        resp = self.client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=oa_msgs,
        )
        return ProviderResponse(text=resp.choices[0].message.content)

    def vision_qa(self, prompt: str, image_path: str) -> ProviderResponse:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        resp = self.client.chat.completions.create(
            model=OPENAI_TEXT_MODEL,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}} 
                    ],
                },
            ],
        )
        return ProviderResponse(text=resp.choices[0].message.content)

    def audio_to_text(self, file_path: str, translate: bool = False) -> ProviderResponse:
        """
        Convert speech audio to text.
        If translate=True, output is English (regardless of input language).
        """
        with open(file_path, "rb") as audio_file:
            if translate:
                transcript = self.client.audio.translations.create(
                    model="whisper-1",
                    file=audio_file
                )
            else:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )

        return ProviderResponse(text=transcript.text)



class GeminiProvider:
    def __init__(self):
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Missing GOOGLE_API_KEY")
        if genai is None:
            raise RuntimeError("google-generativeai package not installed")
        genai.configure(api_key=key)

    def text(self, messages: List[Dict[str, str]]) -> ProviderResponse:
        # Gemini's python lib expects a single prompt string. We send the whole conversation as a prompt.
        # Use the system instruction via system_instruction param.
        conv = []
        for m in messages:
            if m["role"] == "system":
                continue
            role_label = "User" if m["role"] == "user" else "Assistant"
            conv.append(f"{role_label}: {m['content']}")
        prompt = "\n".join(conv)
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL, system_instruction=DEFAULT_SYSTEM_PROMPT)
        resp = model.generate_content(prompt)
        return ProviderResponse(text=resp.text)

    def vision_qa(self, prompt: str, image_path: str) -> ProviderResponse:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        part = {"mime_type": "image/png", "data": b64}
        model = genai.GenerativeModel(GEMINI_TEXT_MODEL)
        resp = model.generate_content([prompt, part])
        return ProviderResponse(text=resp.text)


class GroqProvider:
    def __init__(self):
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("Missing GROQ_API_KEY")
        if GroqClient is None:
            raise RuntimeError("groq package not installed")
        self.client = GroqClient(api_key=key)

    def text(self, messages: List[Dict[str, str]]) -> ProviderResponse:
        # Groq chat-like API usage mirrors OpenAI in this example
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        resp = self.client.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=msgs,
        )
        return ProviderResponse(text=resp.choices[0].message.content)


class StableDiffusionProvider:
    def image_generate(self, prompt: str) -> ProviderResponse:
        if not HF_API_KEY:
            raise RuntimeError("Missing HF_API_KEY")
        url = f"https://api-inference.huggingface.co/models/{STABLE_DIFFUSION_MODEL}"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        resp = requests.post(url, headers=headers, json={"inputs": prompt})
        if resp.status_code != 200:
            raise RuntimeError(f"HF API error: {resp.text}")
        return ProviderResponse(image_bytes=resp.content)

# -------------------------
# Factory
# -------------------------

def get_provider(name: str):
    name = name.lower()
    if name == "openai":
        return OpenAIProvider()
    if name == "gemini":
        return GeminiProvider()
    if name == "groq":
        return GroqProvider()
    if name == "stable":
        return StableDiffusionProvider()
    raise ValueError(f"Unknown provider: {name}")

# -------------------------
# Streamlit App
# -------------------------

def run_streamlit():
    # page config must be called before other Streamlit functions that set layout
    st.set_page_config(page_title="Orion The Ultimate Chatbot", layout="wide")

    # Theme toggle (simple CSS change)
    st.sidebar.title("Choose Theme")
    theme = st.sidebar.radio("", ["🌞 Light", "🌙 Dark"])
    if theme == "🌙 Dark":
        dark_css = """
        <style>
        /* ===== DARK MODE ===== */

        /* page background & main text */
        .stApp, body { background-color: #0e1117 !important; color: #e6eef8 !important; }
        .stApp .block-container { background-color: #0e1117 !important; }

        /* buttons */
        button {
            background-color: #1f2933 !important;
            color: #fff !important;
            border-radius: 20px !important;
            border: 1px solid #444 !important;
        }

        /* input + textarea fields */
        textarea, input {
            background-color: #1f2328 !important;
            color: #e6eef8 !important;
            border-radius: 12px !important;
            border: 1px solid #444 !important;
            padding: 8px 12px !important;
        }

        /* placeholder text visibility */
        textarea::placeholder, input::placeholder {
            color: #aaa !important;
            opacity: 1 !important;
        }

        /* dropdown (selectbox) */
        div[data-baseweb="select"] {
            background-color: #1f2328 !important;
            color: #e6eef8 !important;
            border-radius: 6px !important;   /* 👈 rectangular */
            border: 1px solid #444 !important;
            padding: 6px 10px !important;
            box-shadow: none !important;
        }
        div[data-baseweb="select"] div {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
        }
        div[data-baseweb="select"] * {
            color: #e6eef8 !important;
        }

        /* pill-shaped tabs */
        .stTabs [role="tablist"] { gap: 12px; }
        .stTabs [role="tab"] {
            border: 1px solid #444;
            border-radius: 30px;
            padding: 10px 24px;
            font-size: 15px;
            background-color: #1f2933;
            color: #e6eef8;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        .stTabs [role="tab"]:hover {
            background-color: #2a2f3a;
            transform: scale(1.03);
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #4a90e2;
            color: white;
            border: 1px solid #4a90e2;
            box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        }
        </style>
        """
        st.markdown(dark_css, unsafe_allow_html=True)
    else:
        # Remove previous CSS if switching to light — easiest is to inject light CSS override
        light_css = """
        <style>
        /* ===== LIGHT MODE ===== */

        /* page background & main text */
        .stApp, body { background-color: white !important; color: black !important; }
        .stApp .block-container { background-color: white !important; }
       
        /* buttons */
        button {
            background-color: #f0f0f0 !important;
            color: #000 !important;
            border-radius: 20px !important;
            border: 1px solid #ccc !important;
            padding: 6px 14px !important;
        }

        /* input + textarea fields */
        textarea, input {
            background-color: #fff !important;
            color: #000 !important;
            border-radius: 12px !important;
            border: 1px solid #ccc !important;
            padding: 8px 12px !important;
        }

        /* placeholder text visibility */
        textarea::placeholder, input::placeholder {
            color: #555 !important;
            opacity: 1 !important;
        }

        /* dropdown (selectbox) */
        div[data-baseweb="select"] {
            background-color: #fff !important;
            color: #000 !important;
            border-radius: 12px !important;
            border: 1px solid #ccc !important;
        }
        div[data-baseweb="select"] * {
            color: #000 !important;
        }

        /* make labels and markdown text visible */
        label, .stMarkdown, .stText, .stSelectbox label {
            color: #000 !important;
            font-weight: 500 !important;
        }

        /* pill-shaped tabs */
        .stTabs [role="tablist"] { gap: 12px; }
        .stTabs [role="tab"] {
            border: 1px solid #ccc;
            border-radius: 30px;
            padding: 10px 24px;
            font-size: 15px;
            background-color: #f9f9f9;
            color: black;
            font-weight: 600;
            transition: all 0.2s ease-in-out;
        }
        .stTabs [role="tab"]:hover {
            background-color: #e6e6e6;
            transform: scale(1.03);
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background-color: #4a90e2;
            color: white;
            border: 1px solid #4a90e2;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }
        </style>
        """
        st.markdown(light_css, unsafe_allow_html=True)

    st.title("🤖 🌟 Orion at your Service 🌟 🤖")

    # initialize session state conversation model-aware
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
    if "images" not in st.session_state:
        st.session_state.images = []

    # layout
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "🎨 Image Generation", "👁️ Vision Q&A", "🎙️ Audio → Text/Translate"])

    # --- Chat Tab ---

    with tab1:
        st.subheader("Text Chat")
        provider_choice = st.selectbox("Choose text provider", ["OpenAI", "Gemini", "Groq"], index=0)
        # Initialize chat input if not exists
        if "chat_input_area" not in st.session_state:
            st.session_state.chat_input_area = ""

        # Callback to send message
        def send_message():
            user_text = st.session_state.chat_input_area.strip()
            if not user_text:
                st.warning("Type a message first.")
                return
            provider = get_provider(provider_choice.lower())
            st.session_state.messages.append({"role": "user", "content": user_text, "provider": provider_choice})
            try:
                send_msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                resp = provider.text(send_msgs)
                st.session_state.messages.append({"role": "assistant", "content": resp.text or "<no response>", "provider": provider_choice})
            except Exception as e:
                error_message = str(e)

                # Handle OpenAI quota error
                if "insufficient_quota" in error_message or "429" in error_message:
                    friendly_msg = (
                        "⚠️ OpenAI quota exceeded. Please check your plan/billing, "
                        "or switch to Gemini/Groq from the provider dropdown."
                    )
                else:
                    friendly_msg = f"⚠️ An error occurred: {error_message}"

                # Show in UI
                st.error(friendly_msg)

                # Save clean message in chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": friendly_msg,
                    "provider": provider_choice
                })
                # Clear input properly
            st.session_state.chat_input_area = ""

        # Input field + button
        st.text_area("Type your message", key="chat_input_area")
        st.button("Send", key="chat_send", on_click=send_message)



        # show conversation
        st.markdown("### Chat History")
        for msg in st.session_state.messages:
            if msg["role"] == "system":
                continue
            provider_label = msg.get("provider", "")
            if msg["role"] == "user":
                st.markdown(f"**You ({provider_label})**: {msg['content']}")
            else:
                st.markdown(f"**Assistant ({provider_label})**: {msg['content']}")

        # Download and clear
        chat_log = "\n\n".join(
            [f"You ({m.get('provider','')}) : {m['content']}" if m['role']=='user' else f"Assistant ({m.get('provider','')}) : {m['content']}"
             for m in st.session_state.messages if m['role'] != 'system']
        )
        st.download_button("Download Chat Log", data=chat_log, file_name="chat_log.txt")
        if st.button("Clear History"):
            st.session_state.messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
            st.session_state.images = []
            st.experimental_rerun()

    # --- Image Generation Tab ---

    with tab2:
        st.subheader("Image Generation")
        img_prompt = st.text_area("Image prompt", key="img_prompt")
        if st.button("Generate Image"):
            try:
                sd_provider = StableDiffusionProvider()
                resp = sd_provider.image_generate(img_prompt)
                st.image(resp.image_bytes, caption="Generated Image", use_container_width=True)
                st.session_state.images.append(resp.image_bytes)
            except Exception as e:
                st.error(f"Image generation failed: {e}")

        if st.session_state.images:
            st.markdown("### Generated Images")
            for i, img in enumerate(st.session_state.images):
                st.image(img, width=240)
                st.download_button(f"Download Image {i+1}", data=img, file_name=f"generated_{i+1}.png", mime="image/png")

    
    # --- Vision Q&A Tab ---

    with tab3:
        st.subheader("Vision Q&A")

        if "vision_history" not in st.session_state:
            st.session_state.vision_history = []

        v_provider = st.selectbox("Choose provider", ["OpenAI", "Gemini"], key="vision_provider")
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="vision_upload")
        question = st.text_input("Ask a question about the image", key="vision_question")

        if st.button("Analyze", key="vision_analyze"):
            if uploaded_file is None or question.strip() == "":
                st.warning("Please upload an image and enter a question.")
            else:
                try:
                    tmp_path = f"temp_{uploaded_file.name}"
                    with open(tmp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    provider = get_provider(v_provider.lower())
                    resp = provider.vision_qa(question, tmp_path)

                    # Save to vision history
                    st.session_state.vision_history.append({
                        "provider": v_provider,
                        "question": question,
                        "answer": resp.text,
                        "image_name": uploaded_file.name
                    })

                except Exception as e:
                    error_message = str(e)

                    # Handle quota exceeded error
                    if "insufficient_quota" in error_message or "429" in error_message:
                        friendly_msg = (
                            "⚠️ OpenAI quota exceeded. Please check your plan/billing, "
                            "or switch to Gemini from the provider dropdown."
                     )
                    else:
                        friendly_msg = f"⚠️ Vision Q&A failed: {error_message}"

                    # Show in UI
                    st.error(friendly_msg)

                    # Save clean message in history
                    st.session_state.vision_history.append({
                        "provider": v_provider,
                        "question": question,
                        "answer": friendly_msg,
                        "image_name": uploaded_file.name if uploaded_file else "N/A"
                    })

        # Show history
        if st.session_state.vision_history:
            st.subheader("Vision Q&A History")
            for idx, entry in enumerate(st.session_state.vision_history):
                st.markdown(f"**Question ({entry['provider']}, {entry['image_name']})** : {entry['question']}")
                st.markdown(f"**Answer** : {entry['answer']}")
                st.write("---")

            # Download history
            vision_log = "\n\n".join([f"Question ({h['provider']}, {h['image_name']}): {h['question']}\nA: {h['answer']}" 
                                  for h in st.session_state.vision_history])
            st.download_button("Download Vision Q&A Log", data=vision_log, file_name="vision_log.txt")

            # Clear button
            if st.button("Clear Vision History"):
                st.session_state.vision_history = []
                st.experimental_rerun()

    # --- Audio to Text / Translation Tab ---

    with tab4:
        st.subheader("🎙️ Convert Audio to Text / Translate")
        uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"], key="audio_upload")
        mode = st.radio("Choose mode", ["Transcribe (same language)", "Translate to English"], key="audio_mode")
        if st.button("Process Audio", key="audio_process"):
            if uploaded_audio is None:
                st.warning("Please upload an audio file first.")
            else:
                try:
                    provider = OpenAIProvider()
                    temp_audio_path = f"temp_{uploaded_audio.name}"
                    with open(temp_audio_path, "wb") as f:
                        f.write(uploaded_audio.getbuffer())
                    resp = provider.audio_to_text(temp_audio_path, translate=(mode == "Translate to English"))
                    st.success("Processing complete!")
                    st.text_area("Result", resp.text, height=200)
                except Exception as e:
                    error_message = str(e)

                    # Handle quota exceeded error
                    if "insufficient_quota" in error_message or "429" in error_message:
                        friendly_msg = (
                            "⚠️ OpenAI quota exceeded. Please check your plan/billing, "
                            "or try transcription later."
                        )
                    else:
                        friendly_msg = f"⚠️ Audio processing failed: {error_message}"

                    # Show in UI
                    st.error(friendly_msg)


if __name__ == "__main__":
    run_streamlit()
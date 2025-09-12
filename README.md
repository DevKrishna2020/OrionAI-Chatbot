# 🌟 Orion – Multi-Model AI Chatbot (Streamlit Only)

Orion is a **Multi-Model AI Assistant** built with **Streamlit**.  

It is an excellent Chatbot which integrates multiple providers to handle text, vision, image generation, and audio-to-text/translation.

---

## ✨ Features

- **💬 Text Chat**

  - OpenAI (GPT-4o-mini)
  - Google Gemini (1.5 Flash)
  - Groq (LLaMA-3.3-70B)

- **🎨 Image Generation**

  - Stable Diffusion (via Hugging Face Inference API)

- **👁️ Vision Q&A**

  - Ask questions about images (OpenAI / Gemini)

- **🎙️ Audio → Text/Translate**

  - Speech-to-text and translation (OpenAI)

---

## 🛠️ Installation 

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/orion-chatbot.git
   cd orion-chatbot


2. **Environment Variables**

   Create a .env file in the project root and add:
  
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   GROQ_API_KEY=your_groq_api_key
   HF_API_KEY=your_huggingface_api_key


3. **Run the App**

   streamlit run main_code.py


4. **Project Structure**

.
├── main_code.py       # Main Streamlit app
├── requirements.txt   # Dependencies
├── .env               # API keys (ignored by git)
├── .gitignore         # Hides sensitive files
└── README.md          # Project docs

## 📜License and 👨‍💻Author

MIT License – free to use and modify. – see [LICENSE](LICENSE) file for details.

Developed by **Dev Krishna Pradeep**
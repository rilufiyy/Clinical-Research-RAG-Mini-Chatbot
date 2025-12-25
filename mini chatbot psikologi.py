"""
Chatbot Psikologi - Streaming dengan Groq dan Streamlit
========================================================

Aplikasi chatbot psikologi yang menggunakan:
- Groq API (Llama 3.3) sebagai model bahasa (GRATIS & CEPAT!)
- Streamlit untuk antarmuka pengguna
- LangChain untuk manajemen percakapan dan memori
- SQLite untuk penyimpanan riwayat chat

Fitur:
- Streaming response real-time
- Memory management per user
- Session management
- Interactive UI
- Fokus pada topik Psikologi

Author: Sri Lutfiya Dwiyeni

Cara menjalankan:
streamlit run chat_stream_openai.py
"""

import streamlit as st
import os
from dotenv import load_dotenv

# Import Groq LangChain integration (mengganti OpenAI)
from langchain_groq import ChatGroq

# Import komponen untuk prompt template
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

# Import untuk manajemen riwayat percakapan
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Import output parser
from langchain_core.output_parsers import StrOutputParser

# Load environment variables dari file .env
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Chatbot Psikologi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup judul dan deskripsi aplikasi Streamlit
st.title("🧠 Chatbot Konsultasi Psikologi")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
    <p style='margin: 0; color: #262730;'>
        💡 <strong>Selamat datang!</strong> Saya adalah asisten AI yang dapat membantu Anda dengan pertanyaan seputar psikologi,
        kesehatan mental, dan well-being. Silakan bertanya dengan bebas!
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar untuk user settings
with st.sidebar:
    st.markdown("### 👤 Pengaturan Pengguna")
    user_id = st.text_input("ID Pengguna", "user_001", help="ID unik untuk menyimpan riwayat chat Anda")

    st.markdown("### ℹ️ Informasi")
    st.info("""
    **Topik yang bisa dibahas:**
    - Kesehatan Mental
    - Manajemen Stress & Anxiety
    - Self-care & Well-being
    - Hubungan Interpersonal
    - Psikologi Klinis
    - Terapi & Konseling
    """)

    st.markdown("### ⚠️ Disclaimer")
    st.warning("""
    Chatbot ini hanya untuk informasi edukatif.
    Bukan pengganti konsultasi profesional dengan psikolog atau psikiater.
    """)

def get_session_history(session_id):
    """
    Fungsi untuk mendapatkan riwayat session berdasarkan session_id
    
    Args:
        session_id (str): ID unik untuk setiap session pengguna
    
    Returns:
        SQLChatMessageHistory: Object yang menyimpan riwayat chat dalam database SQLite
    """
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Tombol untuk memulai percakapan baru
# Ketika ditekan, akan menghapus riwayat chat dari UI dan database
if st.button("Mulai Percakapan Baru"):
    st.session_state.chat_history = []  # Hapus dari session state
    history = get_session_history(user_id)
    history.clear()  # Hapus dari database

# Menampilkan riwayat chat yang tersimpan dalam session state
# Loop melalui setiap pesan dan tampilkan dengan role yang sesuai
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

### Setup Model LLM Groq ###
# Konfigurasi Groq LLM
# Menggunakan API key dari environment variable untuk keamanan
api_key = os.getenv('GROQ_API_KEY')

# Validasi API key
if not api_key:
    st.error("⚠️ **GROQ_API_KEY tidak ditemukan!**")
    st.info("""
    **Cara mendapatkan API key GRATIS:**
    1. Kunjungi: https://console.groq.com/keys
    2. Buat akun (gratis, tidak perlu kartu kredit!)
    3. Generate API key
    4. Tambahkan ke file `.env`:
       ```
       GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxx
       ```
    5. Restart aplikasi
    """)
    st.stop()

# Inisialisasi ChatGroq dengan konfigurasi yang sesuai
llm = ChatGroq(
    groq_api_key=api_key,
    model="llama-3.3-70b-versatile",  # Model Groq yang digunakan (GRATIS!)
    temperature=0.7,                   # Tingkat kreativitas untuk respons yang natural (0-1)
    max_tokens=2048,                   # Panjang maksimal respons
    streaming=True                      # Enable streaming untuk respons real-time
)

# Membuat template untuk system message (instruksi untuk AI)
system_prompt = """Anda adalah asisten AI ahli psikologi yang membantu dan empatik.

Peran Anda:
- Memberikan informasi edukatif tentang psikologi dan kesehatan mental
- Mendengarkan dengan empati dan tanpa menghakimi
- Memberikan perspektif yang balanced dan evidence-based
- Menyarankan strategi coping yang sehat

Pedoman Respons:
1. Gunakan bahasa Indonesia yang ramah dan mudah dipahami
2. Berikan respons yang empatik dan supportive
3. Jika topik serius (depresi, anxiety, trauma), sarankan konsultasi profesional
4. Fokus pada psychoeducation dan self-help strategies
5. Jangan memberikan diagnosis medis
6. Hormati privasi dan kerahasiaan pengguna

Ingat: Anda adalah sumber informasi edukatif, bukan pengganti terapis profesional."""

system = SystemMessagePromptTemplate.from_template(system_prompt)

# Template untuk input pengguna
human = HumanMessagePromptTemplate.from_template("{input}")

# Menggabungkan semua komponen prompt:
# 1. System message - instruksi dasar untuk AI
# 2. MessagesPlaceholder - tempat untuk riwayat percakapan
# 3. Human message - input terbaru dari pengguna
messages = [system, MessagesPlaceholder(variable_name='history'), human]

# Membuat ChatPromptTemplate dari komponen-komponen di atas
prompt = ChatPromptTemplate(messages=messages)

# Membuat chain: prompt -> LLM -> output parser
# Chain ini akan memproses input secara berurutan
chain = prompt | llm | StrOutputParser()

# Membungkus chain dengan RunnableWithMessageHistory
# Ini memungkinkan chain untuk mengingat percakapan sebelumnya
runnable_with_history = RunnableWithMessageHistory(
    chain,                          # Chain yang akan dibungkus
    get_session_history,            # Fungsi untuk mendapatkan riwayat session
    input_messages_key='input',     # Key untuk input pesan dalam dictionary
    history_messages_key='history'  # Key untuk riwayat pesan dalam dictionary
)

def chat_dengan_llm(session_id, input_text):
    """
    Fungsi untuk melakukan chat dengan LLM menggunakan streaming
    
    Args:
        session_id (str): ID session untuk mengidentifikasi pengguna
        input_text (str): Teks input dari pengguna
    
    Yields:
        str: Bagian-bagian respons yang di-stream secara real-time
    """
    # Menggunakan stream() untuk mendapatkan respons secara bertahap
    # Ini memungkinkan tampilan real-time seperti ChatGPT
    for output in runnable_with_history.stream(
        {'input': input_text}, 
        config={'configurable': {'session_id': session_id}}
    ):
        yield output

# Input chat dari pengguna
# st.chat_input() membuat input box khusus untuk chat
prompt = st.chat_input("💬 Ketik pertanyaan Anda tentang psikologi di sini...")

# Proses ketika pengguna mengirim pesan
if prompt:
    # Tambahkan pesan pengguna ke riwayat chat
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})

    # Tampilkan pesan pengguna
    with st.chat_message("user"):
        st.markdown(prompt)

    # Tampilkan respons assistant dengan streaming
    with st.chat_message("assistant"):
        # st.write_stream() menampilkan teks secara bertahap (streaming)
        response = st.write_stream(chat_dengan_llm(user_id, prompt))

    # Tambahkan respons assistant ke riwayat chat
    st.session_state.chat_history.append({'role': 'assistant', 'content': response})

import streamlit as st
import cv2
from PIL import Image
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import time
import threading
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Smart Vision Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    div.stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #ddd;
        padding: 10px;
    }
    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: #e9f7ef;
        border-left: 5px solid #4CAF50;
    }
    .output-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        background-color: #f1f8ff;
        border-left: 5px solid #3498db;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: white;
    }
    h1, h2, h3 {
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }
    .stFileUploader {
        padding: 1rem;
        background-color: #f1f8ff;
        border-radius: 10px;
        border: 2px dashed #3498db;
    }
</style>
""", unsafe_allow_html=True)


# API key handling - use Streamlit secrets for secure deployment
def get_api_key():
    # First try to get from secrets.toml
    try:
        return st.secrets["gemini"]["api_key"]
    except:
        # Fall back to the input field if not in secrets
        if "api_key" not in st.session_state:
            st.session_state.api_key = ""

        if not st.session_state.api_key:
            api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")
            if api_key:
                st.session_state.api_key = api_key
                st.sidebar.success("API key saved for this session!")
                return api_key
            else:
                st.sidebar.warning("Please enter a valid API key to use the app")
                return None
        return st.session_state.api_key


# Configure Gemini AI API
api_key = get_api_key()
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Text-to-Speech setup
@st.cache_resource
def get_tts_engine():
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    return engine


tts_engine = get_tts_engine()


def speak(text):
    if text and not st.session_state.get('mute', False):
        threading.Thread(target=_speak, args=(text,)).start()


def _speak(text):
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")


def fetch_lottie_animation(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def listen_for_command():
    recognizer = sr.Recognizer()
    with st.spinner("🎧 Listening..."):
        try:
            mic = sr.Microphone()
            with mic as source:
                st.markdown('<div class="status-box">🎤 Say something... (Adjusting for ambient noise)</div>',
                            unsafe_allow_html=True)
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

            st.markdown('<div class="status-box">🔍 Processing your command...</div>', unsafe_allow_html=True)
            command = recognizer.recognize_google(audio).lower()
            st.success(f"I heard: {command}")
            return command
        except sr.UnknownValueError:
            st.warning("Sorry, I couldn't understand that. Please try again.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
            return None
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return None


def capture_image_from_camera():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Camera access failed. Please check your camera connection.")
            return None, None

        # Allow camera to warm up
        time.sleep(0.5)

        # Capture multiple frames to allow auto-exposure to adjust
        for _ in range(5):
            ret, frame = cap.read()
            time.sleep(0.1)

        # Now capture the final frame
        ret, frame = cap.read()
        cap.release()

        if ret:
            # Apply some basic enhancement
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)  # Slight contrast enhancement

            img_name = "captured_image.jpg"
            cv2.imwrite(img_name, frame)
            return img_name, frame
        else:
            st.error("Failed to capture image.")
            return None, None
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        return None, None


def load_and_prepare_image(image_path):
    try:
        if isinstance(image_path, str):
            im = Image.open(image_path)
        else:
            im = Image.open(image_path)

        # Enhance image quality
        im = im.convert('RGB')
        im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
        return im
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None


def create_prompt_and_generate(image):
    if not api_key:
        st.error("Please enter a valid API key in the sidebar to use this feature")
        return None

    prompt = """
    You are an expert in computer vision. Analyze this image and provide:

    1. A detailed description of what you see (3-4 sentences).
    2. Identify main objects and people in the image.
    3. Note any text visible in the image.
    4. Provide context or setting of the image.
    5. If applicable, suggest how this image might be used or improved.

    Format your response with clear headings and bullet points where appropriate.
    """

    try:
        response = model.generate_content(
            contents=[{"role": "user", "parts": [prompt, image]}],
            generation_config={"temperature": 0.4, "max_output_tokens": 800},
        )
        return response.text
    except Exception as e:
        st.error(f"AI processing error: {str(e)}")
        return "I encountered an error while processing the image. Please try again."


def answer_question(question):
    if not api_key:
        st.error("Please enter a valid API key in the sidebar to use this feature")
        return None

    try:
        # Add some system prompt to make responses higher quality
        enhanced_question = f"""
        Question: {question}

        Please provide a thorough but concise answer. Include relevant facts and examples when helpful.
        Format your response with clear structure, using bullet points or numbered lists if appropriate.
        """

        response = model.generate_content(
            enhanced_question,
            generation_config={"temperature": 0.2, "max_output_tokens": 800},
        )

        # Add to chat history
        st.session_state.chat_history.append({"question": question, "answer": response.text})

        return response.text
    except Exception as e:
        st.error(f"AI processing error: {str(e)}")
        return "I encountered an error while processing your question. Please try again."


# Sidebar
with st.sidebar:
    st.title("🧠 Smart Vision Assistant")

    # Add animation
    lottie_url = "https://lottie.host/ef0c87fa-8e4b-4d63-9e49-ec40932ddcef/c3jXSLmJQq.json"
    lottie_json = fetch_lottie_animation(lottie_url)
    if lottie_json:
        try:
            from streamlit_lottie import st_lottie

            st_lottie(lottie_json, height=200, key="sidebar_animation")
        except:
            st.image("https://cdn-icons-png.flaticon.com/512/4712/4712030.png", width=150)

    st.markdown("### Features")
    st.markdown("✨ **Voice Recognition**")
    st.markdown("✨ **Image Analysis**")
    st.markdown("✨ **AI Chat**")
    st.markdown("✨ **Text-to-Speech**")

    st.markdown("---")

    # Settings
    st.subheader("⚙️ Settings")

    # Only show settings if API key is provided
    if api_key:
        # Voice settings
        st.markdown("##### Voice Options")
        voice_speed = st.slider("Speech Rate", min_value=100, max_value=250, value=160, step=10)
        if voice_speed:
            tts_engine.setProperty('rate', voice_speed)

        mute = st.checkbox("Mute Voice", value=False)
        st.session_state.mute = mute

    # About section
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Smart Vision Assistant uses advanced AI to analyze images and answer questions, powered by Google's Gemini 1.5 Pro model.")
    st.markdown("v1.0.0 © 2025")

    # GitHub link
    st.markdown("---")
    st.markdown("[View Source on GitHub](https://github.com/your-username/smart-vision-assistant)")

# Main content area
if not api_key:
    st.title("🧠 Smart Vision Assistant")
    st.markdown("### Welcome to Smart Vision Assistant")
    st.info("Please enter your Gemini API key in the sidebar to get started.")

    st.markdown("""
    #### How to get a Gemini API key:
    1. Visit [Google AI Studio](https://makersuite.google.com/)
    2. Sign in with your Google account
    3. Go to the API keys section
    4. Create a new API key
    5. Copy and paste it in the sidebar
    """)

    st.markdown("#### Features available once API key is provided:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("##### 💬 Chat")
        st.markdown("Ask questions and get AI-powered responses")
    with col2:
        st.markdown("##### 📷 Image Analysis")
        st.markdown("Analyze images from camera or uploads")
    with col3:
        st.markdown("##### 🎤 Voice Commands")
        st.markdown("Control the app with your voice")

else:
    st.title("🧠 Smart Vision Assistant")
    st.markdown("Your intelligent companion for vision and conversation")

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📷 Image Analysis", "📋 History"])

    with tab1:
        st.markdown("### Ask me anything or give me a voice command")

        col1, col2 = st.columns([4, 1])

        with col1:
            user_input = st.text_input("Type your question here...", key="text_input")

        with col2:
            voice_button = st.button("🎤 Voice", use_container_width=True)

        submit_button = st.button("🚀 Submit", use_container_width=True)

        if voice_button:
            command = listen_for_command()
            if command:
                if "camera" in command or "picture" in command or "photo" in command or "image" in command:
                    st.session_state.activate_camera = True
                    st.experimental_rerun()
                else:
                    with st.spinner("Thinking..."):
                        response = answer_question(command)
                        if response:
                            st.markdown(f'<div class="output-box">{response}</div>', unsafe_allow_html=True)
                            speak(response[:200])  # Read first part of the response

        if submit_button and user_input:
            with st.spinner("Thinking..."):
                response = answer_question(user_input)
                if response:
                    st.markdown(f'<div class="output-box">{response}</div>', unsafe_allow_html=True)
                    speak(response[:200])  # Read first part of the response

    with tab2:
        st.markdown("### Visual Intelligence")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 📷 Camera Capture")
            camera_button = st.button("📸 Capture Image", use_container_width=True)

            # Check if camera activation is requested
            if st.session_state.get('activate_camera', False) or camera_button:
                st.session_state.activate_camera = False  # Reset the flag

                with st.spinner("📷 Accessing camera..."):
                    image_path, frame = capture_image_from_camera()
                    if image_path and frame is not None:
                        st.image(frame, caption="Captured Image", use_column_width=True)

                        with st.spinner("🧠 Analyzing image..."):
                            image = load_and_prepare_image(image_path)
                            if image:
                                output = create_prompt_and_generate(image)
                                if output:
                                    st.markdown(f'<div class="output-box">{output}</div>', unsafe_allow_html=True)
                                    speak("Here's what I found in the image.")
                                    speak(output[:200])  # Read first part of the analysis

        with col2:
            st.markdown("##### 📤 Upload Image")
            uploaded_file = st.file_uploader("Drag and drop or browse an image", type=["jpg", "png", "jpeg"],
                                             key="file_uploader")

            if uploaded_file is not None:
                image = load_and_prepare_image(uploaded_file)
                if image:
                    st.image(image, caption="Uploaded Image", use_column_width=True)

                    with st.spinner("🧠 Analyzing image..."):
                        output = create_prompt_and_generate(image)
                        if output:
                            st.markdown(f'<div class="output-box">{output}</div>', unsafe_allow_html=True)
                            speak("Here's what I found in the image.")
                            speak(output[:200])  # Read first part of the analysis

    with tab3:
        st.markdown("### Conversation History")

        if not st.session_state.chat_history:
            st.info("Your conversation history will appear here.")
        else:
            for i, exchange in enumerate(st.session_state.chat_history):
                st.markdown(f"#### Conversation {i + 1}")
                st.markdown(f"**Q:** {exchange['question']}")
                st.markdown(f"**A:** {exchange['answer']}")
                st.markdown("---")

            if st.button("Clear History", key="clear_history"):
                st.session_state.chat_history = []
                st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and Gemini AI")

if __name__ == "__main__":
    pass  # Streamlit auto-runs the script
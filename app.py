import os
from uuid import uuid4
import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from agno.tools.eleven_labs import ElevenLabsTools
from agno.utils.audio import write_audio_to_file
from agno.utils.log import logger

# Streamlit UI Setup
st.set_page_config(page_title="📰 ➡ 🎙 Blog to Podcast Agent (Local LLM)", page_icon="🎙")
st.title("📰 ➡ 🎙 Blog to Podcast Agent (Local LLM)")

# Sidebar for API Keys
st.sidebar.header("🔑 API Keys")
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API Key", type="password")

# Input: Blog URL
url = st.text_input("Enter the Blog URL:", "")

# Button: Generate Podcast
generate_button = st.button("🎙 Generate Podcast", disabled=not elevenlabs_api_key)

if not elevenlabs_api_key:
    st.warning("Please enter the ElevenLabs API key to enable podcast generation.")

def fetch_blog_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        content = "\n".join(paragraphs)
        return content.strip()
    except Exception as e:
        return f"ERROR: {e}"

# Load summarizer (only once)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

if generate_button:
    if url.strip() == "":
        st.warning("Please enter a blog URL first.")
    else:
        os.environ["ELEVENLABS_API_KEY"] = elevenlabs_api_key
        with st.spinner("Scraping blog and generating podcast... 🎶"):
            blog_content = fetch_blog_content(url)

            if blog_content.startswith("ERROR:"):
                st.error(f"Error fetching blog: {blog_content}")
            elif len(blog_content) < 50:
                st.error("Fetched content is too short. Please check the URL.")
            else:
                try:
                    summarizer = load_summarizer()

                    # Truncate if too long (to avoid token limits)
                    truncated_blog = blog_content[:4000]

                    summary_parts = summarizer(
                        truncated_blog,
                        max_length=250,
                        min_length=100,
                        do_sample=False
                    )

                    summary_text = summary_parts[0]['summary_text']
                    st.subheader("📝 Podcast Script:")
                    st.write(summary_text)

                    # Generate audio with ElevenLabs
                    tools = ElevenLabsTools(
                        voice_id="JBFqnCBsd6RMkjVDRZzb",
                        model_id="eleven_multilingual_v2",
                        target_directory="audio_generations",
                    )

                    audio_data = tools.text_to_audio(summary_text)

                    if audio_data and audio_data[0].base64_audio:
                        save_dir = "audio_generations"
                        os.makedirs(save_dir, exist_ok=True)
                        filename = f"{save_dir}/podcast_{uuid4()}.wav"
                        write_audio_to_file(audio=audio_data[0].base64_audio, filename=filename)

                        st.success("Podcast generated successfully! 🎧")
                        audio_bytes = open(filename, "rb").read()
                        st.audio(audio_bytes, format="audio/wav")

                        st.download_button(
                            label="Download Podcast",
                            data=audio_bytes,
                            file_name="generated_podcast.wav",
                            mime="audio/wav"
                        )
                    else:
                        st.error("No audio was generated.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logger.error(f"Streamlit app error: {e}")

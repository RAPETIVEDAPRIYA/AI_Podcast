import os
from uuid import uuid4
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.eleven_labs import ElevenLabsTools
from agno.utils.audio import write_audio_to_file
from agno.utils.log import logger
import streamlit as st
import requests
from bs4 import BeautifulSoup

# Streamlit Page Setup
st.set_page_config(page_title="📰 ➡ 🎙 Blog to Podcast Agent", page_icon="🎙")
st.title("📰 ➡ 🎙 Blog to Podcast Agent")

# Sidebar: API Keys
st.sidebar.header("🔑 API Keys")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API Key", type="password")

# Check if both keys are provided
keys_provided = all([openai_api_key, elevenlabs_api_key])

# Input: Blog URL
url = st.text_input("Enter the Blog URL:", "")

# Button: Generate Podcast
generate_button = st.button("🎙 Generate Podcast", disabled=not keys_provided)

if not keys_provided:
    st.warning("Please enter both API keys to enable podcast generation.")

def fetch_blog_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        content = "\n".join(paragraphs)
        return content.strip()
    except Exception as e:
        return f"ERROR: {e}"

if generate_button:
    if url.strip() == "":
        st.warning("Please enter a blog URL first.")
    else:
        # Set API keys as environment variables
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["ELEVENLABS_API_KEY"] = elevenlabs_api_key

        with st.spinner("Processing... Scraping blog, summarizing, and generating podcast 🎶"):
            blog_content = fetch_blog_content(url)
            
            if blog_content.startswith("ERROR:"):
                st.error(f"Error fetching blog: {blog_content}")
            elif len(blog_content) < 50:
                st.error("Fetched content is too short. Please check the URL.")
            else:
                try:
                    # Create the agent
                    blog_to_podcast_agent = Agent(
                        name="Blog to Podcast Agent",
                        agent_id="blog_to_podcast_agent",
                        model=OpenAIChat(id="gpt-4o"),
                        tools=[
                            ElevenLabsTools(
                                voice_id="JBFqnCBsd6RMkjVDRZzb",  # Replace with your actual ElevenLabs voice ID
                                model_id="eleven_multilingual_v2",
                                target_directory="audio_generations",
                            )
                        ],
                        description="You are an AI agent that can generate audio using the ElevenLabs API.",
                        instructions=[
                            "Create a concise summary of the provided blog content that is NO MORE than 2000 characters long.",
                            "The summary should capture the main points while being engaging and conversational.",
                            "Use the ElevenLabsTools to convert the summary to audio.",
                            "Ensure the summary stays within 2000 characters to avoid ElevenLabs API limits."
                        ],
                        markdown=True,
                        debug_mode=True,
                    )

                    # Run the agent
                    podcast: RunResponse = blog_to_podcast_agent.run(
                        f"Convert this blog content to a podcast:\n\n{blog_content}"
                    )

                    save_dir = "audio_generations"
                    os.makedirs(save_dir, exist_ok=True)

                    if podcast.audio and len(podcast.audio) > 0:
                        filename = f"{save_dir}/podcast_{uuid4()}.wav"
                        write_audio_to_file(
                            audio=podcast.audio[0].base64_audio,
                            filename=filename
                        )

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
                        st.error("No audio was generated. Please try again.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    logger.error(f"Streamlit app error: {e}")

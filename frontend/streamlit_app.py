import requests
import streamlit as st
from urllib.parse import urlparse, parse_qs
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Development mode
DEV = os.getenv("DEV", "true").lower() in ("true", "1", "yes")

# API Base URL based on development mode
if DEV:
    API_BASE = "http://127.0.0.1:8000"
else:
    API_BASE = "https://morphus-rag-chat.vercel.app"

st.set_page_config(page_title="Podcast RAG", layout="wide")

st.title("🎧 Morphus Chat with Podcasts")
st.caption("Type a question. Get the most relevant clip. Jump right to the moment in YouTube.")

with st.sidebar:
    st.subheader("Ingest")
    st.markdown("**Add YouTube URLs** (one per line)")
    yt_multiline = st.text_area("Paste URLs", height=120, placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ\n...")

    if st.button("Ingest YouTube URLs"):
        urls = [u.strip() for u in yt_multiline.splitlines() if u.strip()]
        if urls:
            try:
                r = requests.post(f"{API_BASE}/ingest/youtube", json={"urls": urls}, timeout=120)

                # Check if the request was successful
                if r.status_code == 200:
                    payload = r.json()

                    # Check for errors in the response
                    errors = [x for x in payload.get("results", []) if x.get("status") == "error"]

                    if errors:
                        # Display errors with details
                        st.error("Some URLs could not be processed:")
                        for error in errors:
                            url = error.get("url", "Unknown URL")
                            error_type = error.get("error_type", "unknown_error")
                            details = error.get("error_details", "Unknown error")

                            with st.expander(f"Error processing: {url}"):
                                st.write(f"**Error type:** {error_type}")
                                st.write(f"**Details:** {details}")
                                if error.get("error"):
                                    st.code(error.get("error"), language="text")

                        # If some URLs were successful, show those too
                        successes = [x for x in payload.get("results", []) if x.get("status") == "ok"]
                        if successes:
                            st.success(f"Successfully processed {len(successes)} out of {len(payload.get('results', []))} URLs.")
                    else:
                        st.success(f"All {len(payload.get('results', []))} URLs processed successfully!")
                else:
                    # Handle HTTP error responses
                    try:
                        error_data = r.json()
                        st.error(f"Error: {error_data.get('detail', {}).get('message', 'Failed to process URLs')}")

                        # Display detailed errors if available
                        errors = error_data.get('detail', {}).get('errors', [])
                        if errors:
                            for error in errors:
                                url = error.get("url", "Unknown URL")
                                error_type = error.get("error_type", "unknown_error")
                                details = error.get("error_details", "Unknown error")

                                with st.expander(f"Error processing: {url}"):
                                    st.write(f"**Error type:** {error_type}")
                                    st.write(f"**Details:** {details}")
                    except:
                        # If we can't parse the JSON response
                        st.error(f"Error: HTTP {r.status_code} - {r.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {str(e)}")
        else:
            st.warning("No URLs provided.")


def yt_id_from_url(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    if not url:
        return None

    try:
        parsed = urlparse(url)
        # Handle youtu.be URLs
        if parsed.hostname in ("youtu.be",):
            return parsed.path.strip("/")
        # Handle youtube.com URLs
        if parsed.hostname and "youtube.com" in parsed.hostname:
            qs = parse_qs(parsed.query)
            return qs.get("v", [None])[0]
        # Handle youtube.com/embed URLs
        if parsed.hostname and "youtube.com" in parsed.hostname and "/embed/" in parsed.path:
            return parsed.path.split("/embed/")[1].split("/")[0].split("?")[0]
    except Exception:
        return None
    return None

def to_watch_url_with_start(url: str, start: int) -> str:
    """
    Convert a YouTube URL to a watch URL with start time parameter.
    This format works well with st.video for embedding.
    """
    if not url:
        return url

    # Get the video ID
    vid = yt_id_from_url(url)
    if not vid:
        return url

    # Create a watch URL with start time
    return f"https://www.youtube.com/embed/{vid}?start={start}&autoplay=0"

q = st.text_input("Ask about your podcast:", placeholder="e.g., What did she say about magnesium and sleep?")

# Add a caption to explain what the user is seeing
st.caption("Enter a query and click Search to see the single best-matching timestamped video segment.")

if st.button("Search") and q:
    with st.spinner("Searching..."):
        try:
            r = requests.get(f"{API_BASE}/search", params={"q": q}, timeout=60)

            if r.status_code == 200:
                data = r.json()
                result = data.get("results", [])
            else:
                # Handle HTTP error responses
                try:
                    error_data = r.json()
                    st.error(f"Error: {error_data.get('detail', 'Search failed')}")
                except:
                    # If we can't parse the JSON response
                    st.error(f"Error: HTTP {r.status_code} - {r.text}")
                # Return early as we have no results to display
                st.stop()
        except Exception as e:
            st.error(f"Error connecting to backend: {str(e)}")
            st.stop()

    # If we get here, we should have results to display
    if not result:
        st.info("No results found for your query. Try a different question or ingest more content.")
        st.stop()

    result = result[0]
    if result:
        title = result.get("title") or "Untitled"
        snippet = result.get("snippet") or ""
        url = result.get("video_url")
        start = int(result.get("start_seconds") or 0)
        start_hms = result.get("start_hms") or "00:00"

        st.subheader(title)
        # Use the score from server-side reranking
        score = result.get('score', 0.0)
        st.caption(f"Jump at **{start_hms}** • score={score:.3f}")

        # Display the AI response if available
        ai_response = result.get('ai_response')
        if ai_response:
            st.markdown(f"**Why this is relevant:**")
            st.markdown(ai_response)
            st.markdown("---")

        if url:
            # Use the helper function to create a watch URL with start time
            watch_url = to_watch_url_with_start(url, start)

            # Use st.video for full-width YouTube embedding
            st.components.v1.iframe(watch_url, width = 1000, height=480)

            # Add a direct link below the video
            st.markdown(f"[Open in YouTube ↗]({watch_url})")
        else:
            st.info("No video URL associated with this chunk.")

        with st.expander("Original Transcript"):
            st.write(snippet)
    else:
        st.info("No results available.")

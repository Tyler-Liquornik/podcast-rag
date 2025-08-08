import os
import requests
import streamlit as st
from urllib.parse import urlparse, parse_qs
from typing import Optional

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Podcast RAG (YouTube jump)", layout="wide")

st.title("🎧 Podcast RAG with Timestamp Jump")
st.caption("Type a question. Get relevant clips. Jump right to the moment in YouTube.")

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

    st.markdown("---")
    st.markdown("**Clear All Data**")

    clear_col1, clear_col2 = st.columns(2)
    with clear_col1:
        clear_button = st.button("Clear Index", type="primary", help="This will delete all data from the Pinecone index")

    if clear_button:
        st.warning("⚠️ This will delete ALL data from the index.")
        confirm_col1, confirm_col2 = st.columns(2)
        with confirm_col1:
            if st.button("Yes, Clear Everything", type="primary"):
                try:
                    with st.spinner("Clearing index..."):
                        r = requests.post(f"{API_BASE}/clear-index", timeout=30)

                    if r.status_code == 200:
                        st.success("Successfully cleared all data from the index!")
                    else:
                        # Handle HTTP error responses
                        try:
                            error_data = r.json()
                            st.error(f"Error: {error_data.get('detail', {}).get('message', 'Failed to clear index')}")
                        except:
                            # If we can't parse the JSON response
                            st.error(f"Error: HTTP {r.status_code} - {r.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend: {str(e)}")


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

q = st.text_input("Ask about your podcast:", placeholder="e.g., What did she say about magnesium and sleep?")
k = st.slider("Results", 1, 12, 6, 1)

if st.button("Search") and q:
    with st.spinner("Searching..."):
        try:
            r = requests.get(f"{API_BASE}/search", params={"q": q, "k": k}, timeout=60)

            if r.status_code == 200:
                data = r.json()
                results = data.get("results", [])
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
    if not results:
        st.info("No results found for your query. You need to ingest some YouTube URLs first.")
        st.stop()

    cols = st.columns(3, gap="large")

    for idx, item in enumerate(results):
        with cols[idx % 3]:
            title = item.get("title") or "Untitled"
            snippet = item.get("snippet") or ""
            url = item.get("video_url")
            start = int(item.get("start_seconds") or 0)
            start_hms = item.get("start_hms") or "00:00"

            st.markdown(f"**{title}**")
            score = item.get('score', 0.0)
            st.caption(f"Jump at **{start_hms}** • score={score:.3f}")
            if url:
                vid = yt_id_from_url(url)
                if vid:
                    embed = f"https://www.youtube.com/embed/{vid}?start={start}&autoplay=0"
                    st.components.v1.iframe(embed, height=315)
                # Determine if the URL already has query parameters
                timestamp_separator = "&" if "?" in url else "?"
                st.link_button("Open on YouTube ↗", f"{url}{timestamp_separator}t={start}s")
            else:
                st.info("No video URL associated with this chunk.")

            with st.expander("Snippet"):
                st.write(snippet)

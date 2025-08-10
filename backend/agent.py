from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from settings import logger, OPENAI_API_KEY

# Initialize the ChatOpenAI model
chat_model = ChatOpenAI(
    model="gpt-5-2025-08-07",
    temperature=1,
    api_key=OPENAI_API_KEY
)

def generate_response(query: str, title: str, snippet: str) -> str:
    """
    Generate a helpful response to the user's query based on the video snippet.

    Args:
        query: The user's search query
        title: The title of the video
        snippet: The transcript snippet from the video

    Returns:
        A generated response that answers the query and explains the context
    """
    system_prompt = """
    You are a helpful AI assistant that explains video content to users. Your task is to:

    1. Generate a concise, helpful response that directly answers the user's question based on the video snippet.
    2. Explain the context of what's happening in the video based on the snippet.
    3. Explain why this specific video segment is relevant to the user's question.

    Use both the video title and the content of the snippet to provide a comprehensive response.
    Keep your response conversational, informative, and under 150 words.

    DO NOT mention that you're an AI or that you're analyzing a transcript.
    DO NOT apologize or use phrases like "Based on the snippet provided".
    DO speak as if you're explaining why this video segment answers their question.
    """

    human_prompt = f"""
    User question: {query}

    Video title: {title}

    Video snippet: {snippet}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    try:
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I couldn't generate a response for this result. Please check the snippet for relevant information."
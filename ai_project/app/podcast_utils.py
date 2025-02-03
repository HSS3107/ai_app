import os
import hashlib
import logging
import re
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import traceback
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import LLMChain, create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from collections import Counter
import tiktoken
from sentence_transformers import SentenceTransformer, util
import torch
from langchain.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import numpy as np
from datetime import datetime
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
import json
import logging
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .models import ChatSession
from django.apps import apps








# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_environment_variables():
    load_dotenv()
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY_1', 'PINECONE_API_ENV', 'PINECONE_INDEX_NAME', 'YOUTUBE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def get_youtube_service():
    return build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))

def get_video_info(video_id):
    youtube = get_youtube_service()
    try:
        request = youtube.videos().list(
            part='snippet,contentDetails',
            id=video_id
        )
        response = request.execute()
        
        if 'items' in response:
            item = response['items'][0]
            snippet = item['snippet']
            content_details = item['contentDetails']
            return {
                'title': snippet['title'],
                'channel': snippet['channelTitle'],
                'published_at': snippet['publishedAt'],
                'description': snippet['description'],
                'duration': content_details['duration'],
                'captions': content_details.get('caption', 'false') == 'true'
            }
    except HttpError as e:
        logger.error(f"An error occurred: {e}")
    return None

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return transcript
    except Exception as e:
        logger.error(f"An error occurred while fetching the transcript for video ID {video_id}: {str(e)}")
        return None

# Create YouTube service object
youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
 
def get_video_stats(video_id):
    request = youtube.videos().list(
        part="statistics",
        id=video_id
    )
    response = request.execute()
    
    stats = response['items'][0]['statistics']
    return {
        "views": stats.get('viewCount', 0),
        "likes": stats.get('likeCount', 0),
        "dislikes": stats.get('dislikeCount', 0),
    }
    
# Function to get top comments
def get_top_comments(video_id, max_results=5):
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        order="relevance"
    )
    response = request.execute()

    comments = []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append({
            "author": comment['authorDisplayName'],
            "text": comment['textDisplay'],
            "likes": comment['likeCount'],
        })

    return comments

# used in main code
def format_original_transcript(transcript, video_info, video_id):
    try:
        # Get video statistics and top comments
        stats = get_video_stats(video_id)
        top_comments = get_top_comments(video_id)
        
        formatted_transcript = f"Title: {video_info.get('title', 'Unknown')}\n"
        formatted_transcript += f"Channel: {video_info.get('channel', 'Unknown')}\n"
        formatted_transcript += f"Published: {video_info.get('published_at', 'Unknown')}\n"
        formatted_transcript += f"Duration: {video_info.get('duration', 'Unknown')}\n"
        formatted_transcript += f"Description: {video_info.get('description', 'No description available')}\n\n"
        formatted_transcript += "Transcript:\n\n"
        
        for entry in transcript:
            start_time = format_timecode(entry['start'])
            formatted_transcript += f"[{start_time}] {entry['text']}\n"
        
        return formatted_transcript
    except Exception as e:
        logger.error(f"Error formatting transcript: {str(e)}")
        return "Error: Unable to format transcript"


def format_timecode(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def sanitize_filename(filename):
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c in ' -_.']).rstrip()

def create_embeddings():
    try:
        return OpenAIEmbeddings()
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise

def initialize_pinecone():
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY_1'))
    index_name = os.getenv('PINECONE_INDEX_NAME')
    
    try:
        index = pc.Index(index_name)
        return index
    except Exception as e:
        logger.error(f"Error connecting to Pinecone index: {str(e)}")
        raise
    


def num_tokens_from_string(string: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def format_transcript_chunk_with_gpt35(transcript_chunk, video_info, is_first_chunk=False, is_last_chunk=False):
    gpt35_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=1000)
    
    prompt = PromptTemplate.from_template("""
    Format transcript chunk:
    1. {metadata_instruction}
    2. each line of the transcript should follow the format:  [HH:MM:SS] Speaker: Statement (Sentiment). for example: [00:02:51] Young Zhao: I spent about four years expanding that agency. (Neutral)
    4. {continuation_instruction}
    
    Rules:
    1. Do not summarize or paraphrase. Use the exact words from the original.
    2. Include ALL timestamps and content from the chunk.
    3. identify speaker. in rare case if speaker cannot be identified with above high confidence, then use spaker only. 
    4. Add sentiment at the end of each line.
    5. Do not add any additional commentary or text not present in the original.

    Video: {title}
    Channel: {channel}

    Chunk:
    {transcript_chunk}

    Formatted:
    """)

    metadata_instruction = "Include video metadata." if is_first_chunk else "Continue formatting."
    continuation_instruction = "This is the last chunk." if is_last_chunk else "This chunk continues in the next part."

    while True:
        formatted_prompt = prompt.format(
            metadata_instruction=metadata_instruction,
            continuation_instruction=continuation_instruction,
            title=video_info['title'],
            channel=video_info['channel'],
            transcript_chunk=transcript_chunk
        )
        if num_tokens_from_string(formatted_prompt) <= 3000:
            break
        transcript_chunk = transcript_chunk[:int(len(transcript_chunk)*0.9)]

    try:
        result = gpt35_model.invoke(formatted_prompt)
        return result.content.strip()
    except Exception as e:
        logger.error(f"Error in format_transcript_chunk_with_gpt35: {str(e)}")
        return f"Error formatting chunk: {str(e)}"

def create_ai_formatted_transcript(original_transcript, video_info):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(original_transcript)
    
    formatted_chunks = []
    for i, chunk in enumerate(chunks):
        is_first_chunk = (i == 0)
        is_last_chunk = (i == len(chunks) - 1)
        formatted_chunk = format_transcript_chunk_with_gpt35(chunk, video_info, is_first_chunk, is_last_chunk)
        formatted_chunks.append(formatted_chunk)
    
    return "\n\n".join(formatted_chunks)

def create_or_load_vector_store(texts, embeddings, index, video_id):
    try:
        vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
        
        vector_ids = []
        metadatas = []
        for i, text in enumerate(texts):
            chunk_id = hashlib.md5(f"{video_id}_{i}".encode()).hexdigest()
            vector_ids.append(chunk_id)
            
            timestamp_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', text)
            timestamp = timestamp_match.group(1) if timestamp_match else None
            
            if timestamp:
                metadatas.append({
                    "video_id": video_id,
                    "chunk_index": i,
                    "timestamp": timestamp
                })
                vector_store.add_texts([text], metadatas=[metadatas[-1]], ids=[chunk_id])
        
        logger.info(f"Added {len(metadatas)} valid transcript chunks to vector store for video ID: {video_id}")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating or loading vector store for video_id {video_id}: {str(e)}")
        raise
    


logger = logging.getLogger(__name__)


from typing import Sequence, TypedDict, Annotated
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

# Define state schema for conversation
class ConversationState(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str
    video_title: str  # Added
    content_map: str  # Added

class ChatHistoryManager:
    def __init__(self):
        self._histories: dict[str, list[BaseMessage]] = {}
    
    def get_history(self, video_id: str) -> list[BaseMessage]:
        if video_id not in self._histories:
            self._histories[video_id] = []
        return self._histories[video_id]
    
    def add_message(self, video_id: str, message: BaseMessage):
        if video_id not in self._histories:
            self._histories[video_id] = []
        self._histories[video_id].append(message)
    
    def clear_history(self, video_id: str):
        if video_id in self._histories:
            self._histories[video_id] = []
    
    def get_all_histories(self) -> dict[str, list[BaseMessage]]:
        return self._histories

# Create a global instance
chat_manager = ChatHistoryManager()

def setup_llm_chain(retriever):
    logger.info(f"\n{'='*100}")
    logger.info("INITIALIZING LLM CHAIN")
    logger.info(f"{'='*100}")
    
    # Configure retriever
    retriever.search_kwargs.update({
        "k": 5,  # Increase number of retrieved chunks
        "fetch_k": 10,  # Fetch more candidates for reranking
        "score_threshold": 0.5,  # Minimum similarity score
    })
    
    # Log retriever configuration
    logger.info("\n[RETRIEVER CONFIGURATION]")
    logger.info(f"Retriever type: {type(retriever).__name__}")
    logger.info(f"Retriever details: {retriever}")
    
    # Define and log prompts
    logger.info("\n[PROMPT TEMPLATES]")
    
    contextualize_q_system_prompt = """
    Given a chat history and the latest user question which might reference context in the chat history,
    formulate a standalone question which can be understood without the chat history.
    DO NOT answer the question, just reformulate it if needed and otherwise return it as is.
    """
    logger.info("\n1. Contextualization System Prompt:")
    logger.info(contextualize_q_system_prompt)

    qa_system_prompt = """
    You are an AI assistant specializing in analyzing and answering questions about YouTube videos. 
    You're currently focusing on a video titled "{video_title}".
    
    Use the following content map, relevant transcript context, and conversation history to provide a detailed and accurate answer:

    Content Map:
    {content_map}

    Relevant Transcript Context:
    {context}

    When answering, please:
    1. Use exact timestamp--(which is in the format like [HH:MM:SS] for example: [00:03:47]) when referencing specific moments in the video.
    2. Mention relevant names when discussing their contributions or statements.
    3. Keep the conversation style and humour as that of elon musk. keep the answers terse.
    4. Assume the user is familiar with the topic but may not be an expert.
    5. Some user questions may require knowledge external to the video. Give user a complete answer using external knowledge but also mention that the insight is based on intelligence and knowledge other than that of video. 
    6. Consider the conversation history when providing context and avoiding repetition.
    """
    logger.info("\n2. QA System Prompt:")
    logger.info(qa_system_prompt)

    # Create prompt templates
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # Initialize LLM
    logger.info("\n[LLM CONFIGURATION]")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    logger.info(f"Model: {llm.model_name}")
    logger.info(f"Temperature: {llm.temperature}")
    
    # Create history aware retriever
    logger.info("\n[CREATING HISTORY-AWARE RETRIEVER]")
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    logger.info("History-aware retriever created with components:")
    logger.info(f"- Base Retriever: {type(retriever).__name__}")
    logger.info(f"- LLM: {type(llm).__name__}")
    
    # Create document chain
    logger.info("\n[CREATING DOCUMENT CHAIN]")
    doc_chain = create_stuff_documents_chain(
        llm,
        qa_prompt
    )
    logger.info("Document chain created")
    
    # Create retrieval chain
    logger.info("\n[CREATING FINAL RETRIEVAL CHAIN]")
    retrieval_chain = create_retrieval_chain(history_aware_retriever, doc_chain)
    logger.info("Final retrieval chain assembled with components:")
    logger.info(f"- History-aware retriever: {type(history_aware_retriever).__name__}")
    logger.info(f"- Document chain: {type(doc_chain).__name__}")
    
    logger.info(f"\n{'='*100}")
    return retrieval_chain

# def process_query(retrieval_chain, query, video_id, video_title, content_map, chat_history):
#     try:
#         # Get the conversation history for this video
#         messages = chat_manager.get_history(video_id)
        
#         # Prepare the input with all necessary context
#         chain_input = {
#             "input": query,
#             "chat_history": messages,
#             "video_title": video_title,
#             "content_map": content_map
#         }
        
#         # Log the input for debugging
#         logger.debug(f"Chain input for video {video_id}: {chain_input}")
        
#         # Execute the chain
#         response = retrieval_chain.invoke(chain_input)
        
#         # Extract the answer
#         answer = response.get('answer', "I couldn't generate a response.")
        
#         # Update the chat history
#         chat_manager.add_message(video_id, HumanMessage(content=query))
#         chat_manager.add_message(video_id, AIMessage(content=answer))
        
#         # Update the chat_history list for backward compatibility
#         chat_history.append({"role": "user", "content": query})
#         chat_history.append({"role": "assistant", "content": answer})
        
#         # Log the result for debugging
#         logger.debug(f"Generated answer for video {video_id}: {answer}")
        
#         return answer, chat_history

#     except Exception as e:
#         logger.error(f"Error processing query for video_id {video_id}: {str(e)}")
#         logger.error(traceback.format_exc())
#         return f"An error occurred while processing your query: {str(e)}", chat_history

def create_conversation_graph(retrieval_chain, video_title, content_map):
    # Define the node that processes the conversation
    def process_conversation(state: ConversationState):
        # Call the retrieval chain with all necessary context
        response = retrieval_chain.invoke({
            "input": state["input"],
            "chat_history": state["chat_history"],
            "video_title": state["video_title"],
            "content_map": state["content_map"]
        })
        
        # Return updated state while preserving video metadata
        return {
            "chat_history": [
                HumanMessage(content=state["input"]),
                AIMessage(content=response["answer"])
            ],
            "context": response.get("context", ""),
            "answer": response["answer"],
            "video_title": state["video_title"],
            "content_map": state["content_map"]
        }

    # Create the graph
    workflow = StateGraph(state_schema=ConversationState)
    
    # Add the single conversation node
    workflow.add_node("conversation", process_conversation)
    workflow.add_edge(START, "conversation")

    # Compile the graph with memory
    memory = MemorySaver()
    conversation_app = workflow.compile(checkpointer=memory)
    
    return conversation_app



def create_history_aware_retriever_chain(vector_store, llm, contextualize_q_prompt):
    retriever = vector_store.as_retriever()
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    return history_aware_retriever

def create_conversational_rag_chain(history_aware_retriever, llm, qa_prompt):
    doc_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, doc_chain)
    return retrieval_chain

def get_message_history():
    return []

video_summaries = {}

def store_summary(video_id, content_map):
    video_summaries[video_id] = {
        'content_map': content_map
    }
    logger.info(f"Stored content map for video ID {video_id}")
    
def get_summary(video_id):
    summary_data = video_summaries.get(video_id, {
        'content_map': "No content map available for this video."
    })
    logger.info(f"Retrieved content map for video ID {video_id}")
    return summary_data['content_map']

def generate_comprehensive_summary(ai_formatted_transcript, video_info):
    gpt_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = PromptTemplate.from_template("""
    Create a comprehensive summary of this video transcript. The summary should:
    1. Start with an introduction mentioning the video title, channel, and main topic.
    2. Include main points discussed in the video, with timestamps if available.
    3. Highlight any key quotes or memorable moments.
    4. Identify the overall tone or sentiment of the video.
    5. Conclude with the main takeaway or call to action, if any.
    6. Be under 250 words long.

    Video: {title}
    Channel: {channel}

    Transcript:
    {ai_formatted_transcript}

    Comprehensive Summary:
    """)
#
    try:
        result = gpt_model.invoke(prompt.format(
            title=video_info['title'],
            channel=video_info['channel'],
            ai_formatted_transcript=ai_formatted_transcript[:15000]
        ))
        return result.content.strip()
    except Exception as e:
        logger.error(f"Error generating comprehensive summary: {str(e)}")
        return "Error generating summary."
    


import traceback

def generate_content_map(video_id, video_info):
    
    transcript = get_transcript(video_id)
    
    formatted_transcript = format_original_transcript(transcript, video_info, video_id) if transcript and video_info else "Transcript not available."
    
    gpt_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    
    
    prompt = PromptTemplate.from_template("""
    You are an advanced AI called Diveinyt (Dive-in-it), with expertise in content analysis, data visualization, and pattern recognition. Your task is to generate a comprehensive, detailed and highly valuable analysis of the provided YouTube video transcript and use exact timestamp--(which is in the format like [HH:MM:SS] for example: [00:03:47]) when referencing specific moments in the video.. This analysis should capture every segment of the video, uncover hidden patterns, subtle nuances, and profound insights, including elements that may not have been consciously intended by the creator. Incorporate precise timestamps and sufficient quotes from the speakers to reduce dependency on the overall transcript context. Ensure the analysis is practical, actionable, and considers the diverse needs of all potential user personas. Keep the conversation style and humour as that of elon musk. keep the answers terse.

Begin with a concise summary of the video's content in exactly 50 words, capturing the core message and main points. Follow this with a comprehensive list of relevant tags, each prefixed with a '#', that reflect the key themes, topics, and concepts discussed in the video.

Start the introduction with a compelling opening that immediately grabs attention, such as an intriguing fact or question related to the video's content. Provide a brief overview of the video's main topic, explaining its significance and relevance within a broader context.

Develop an exhaustive content map by breaking down the transcript into its main topics, subtopics, and themes. For each segment, include precise start and end timestamps, a clear and descriptive section title, a detailed description of the content discussed and speakers talking in the segment. Highlight all essential points, arguments, or ideas presented within each segment. For complex sections, further divide them into subtopics with corresponding timestamps and detailed explanations. Incorporate 2-3 significant and precise speaker quotes from the transcript within each segment along with timestamp to enhance context and understanding.

Perform data-driven insights by conducting a sentiment analysis to evaluate the emotional tone throughout the video, noting any shifts with corresponding timestamps. Identify and quantify the most frequently used words and key phrases, discussing their relevance to the video's themes. Predict audience retention and engagement points based on content intensity, topic interest, and presentation style, highlighting sections likely to keep viewers most engaged with precise timestamps. Provide a proportional analysis of time allocated to each topic, expressed as percentages, supported by timestamps.

Map the emotional and intellectual impact by outlining the emotional trajectory of the video, indicating peaks and troughs in emotional intensity with accurate timestamps. Highlight key moments that have significant emotional or intellectual impact, explaining their significance and potential effect on the viewer.

Present key insights and 'Did You Know?' facts by uncovering profound and unexpected findings supported by data, referencing specific timestamps. Compile a list of interesting and lesser-known facts from the video, formatted as 'Did You Know?' statements, including the relevant timestamps. Quantify insights wherever possible to enhance credibility and impact.

Provide contextual background information that enhances the understanding of the video's content. Explain any technical terms, historical references, or contextual details that may not be immediately apparent to all viewers, ensuring the analysis is accessible to a wide audience.

Critically evaluate the video by assessing its strengths and weaknesses using quantifiable measures. Analyze the pacing to determine if it maintains viewer interest throughout, citing timestamps for any pacing issues. Evaluate the complexity of the language used, including vocabulary and sentence structure, and discuss its suitability for the intended audience. Assess the clarity and coherence of the content delivery and the logical structure of the arguments presented. Incorporate metrics such as speech rate (words per minute) to support your evaluation.

Extract notable quotes and excerpts from the transcript that illustrate important points or themes, including precise timestamps. Provide a brief analysis for each quote, explaining its relevance and impact on the overall message.

Analyze the visual and auditory elements of the video by evaluating the use and effectiveness of graphics, charts, and animations. Assess the quality and impact of auditory elements, including background music, sound effects, and the speaker's vocal delivery. Examine how the pacing of visual and auditory elements aligns with the content's overall pacing, using timestamps to highlight specific examples where these elements enhance or detract from viewer engagement and comprehension.

Conclude with a strong summary that encapsulates the essence of the video. Provide a memorable takeaway that reinforces the key insights from your analysis, leaving the reader with a lasting impression or a thought-provoking question that underscores the significance of the content.

Ensure that your analysis is meticulous and exhaustive, covering every aspect of the video without omission. Use precise timestamps and data-driven metrics to support your observations and conclusions. Consider the diverse needs and perspectives of all potential users, making the content accessible, informative, and valuable to a wide audience, including casual viewers, content creators, educators, and subject matter experts. Focus on insights that can be directly applied or considered by viewers and stakeholders, highlighting how the analysis can inform future content development or deepen understanding of the subject matter. Verify all data and metrics for accuracy and reliability, ensuring quotes, timestamps, and references are precise and correctly attributed. Maintain a professional and objective tone throughout, supporting all assertions with evidence and avoiding personal bias. Incorporate charts, graphs, or other visual aids to represent data where applicable, ensuring visuals are clear, accurately labeled, and enhance the reader's understanding of the analysis.

Deliver an unparalleled analysis that serves as an indispensable guide to the video's content. By meticulously examining every element and presenting your findings with precision and depth, you will elevate the reader's understanding and provide valuable insights that resonate across various user personas. Your comprehensive content map, enriched with strategic quotes and data-driven metrics, will set a new standard for content analysis excellence.

    Video: {title}
    Channel: {channel}

    Transcript:
    {formatted_transcript}

    Content Map:
    """)

    try:
        result = gpt_model.invoke(prompt.format(
            title=video_info['title'],
            channel=video_info['channel'],
            formatted_transcript= formatted_transcript
        ))
        return result.content.strip()
    except Exception as e:
        logger.error(f"Error generating content map: {str(e)}")
        return "Error generating content map."
    
def store_summary_and_tags(video_id, summary_and_tags):
    if video_id not in video_summaries:
        video_summaries[video_id] = {}
    video_summaries[video_id]['summary_and_tags'] = summary_and_tags
    logger.info(f"Stored summary and tags for video ID {video_id}")

def load_vector_store(video_id):
    vector_store_path = f"vector_stores/{video_id}"
    if os.path.exists(vector_store_path):
        embeddings = create_embeddings()
        try:
            return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        except ValueError as e:
            logger.error(f"Error loading vector store for video ID {video_id}: {str(e)}")
            return None
    return None



def process_video(video_id):
    logger.info(f"Starting to process video ID: {video_id}")
    try:
        # Get video info
        video_info = get_video_info(video_id)
        if not video_info:
            logger.error(f"Failed to get video info for video ID: {video_id}")
            return None, None, None, None, None, None

        # Setup file paths
        safe_title = sanitize_filename(video_info['title'])
        transcript_path = f"transcripts/{safe_title}.txt"
        content_map_path = f"content_maps/{safe_title}_content_map.txt"
        vector_store_path = f"vector_stores/{video_id}"

        # Check if files exist
        if os.path.exists(transcript_path) and os.path.exists(content_map_path) and os.path.exists(vector_store_path):
            logger.info(f"Transcript, content map, and vector store found for video ID: {video_id}. Loading from files.")
            
            # Load existing files
            with open(transcript_path, 'r', encoding='utf-8') as f:
                original_transcript = f.read()
            with open(content_map_path, 'r', encoding='utf-8') as f:
                content_map = f.read()
            
            # Load vector store
            vector_store = load_vector_store(video_id)
            if vector_store is None:
                logger.error(f"Failed to load vector store for video ID: {video_id}")
                return None, None, None, None, None, None
                
            logger.info(f"Loaded existing vector store for video ID: {video_id}")
        else:
            # Process new video
            transcript = get_transcript(video_id)
            if not transcript:
                logger.error(f"Failed to get transcript for video ID: {video_id}")
                return None, None, None, None, None, None

            logger.info(f"Transcript retrieved for video ID: {video_id}. Length: {len(transcript)}")
            original_transcript = format_original_transcript(transcript, video_info, video_id)
            
            # Save transcript
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(original_transcript)

            # Generate and save content map
            content_map = 'null'
            os.makedirs(os.path.dirname(content_map_path), exist_ok=True)
            with open(content_map_path, 'w', encoding='utf-8') as f:
                f.write(content_map)

            # Create and save vector store
            try:
                embeddings = create_embeddings()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                original_chunks = text_splitter.split_text(original_transcript)
                vector_store = FAISS.from_texts(
                    original_chunks, 
                    embeddings, 
                    metadatas=[{"video_id": video_id} for _ in original_chunks]
                )
                
                os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
                vector_store.save_local(vector_store_path)
                logger.info(f"Created and saved new vector store for video ID: {video_id}")
            except Exception as e:
                logger.error(f"Error creating vector store for video ID {video_id}: {str(e)}")
                return None, None, None, None, None, None

        # Store content map in memory
        store_summary(video_id, content_map)

        # Set up the retrieval chain
        retriever = vector_store.as_retriever(
            search_type="similarity",  # or "mmr" for maximum marginal relevance
            search_kwargs={"k": 4}     # specify number of top results to return
            )
        retrieval_chain = setup_llm_chain(retriever)

        # Initialize conversation app for this video
        conversation_app = create_conversation_graph(
            retrieval_chain,
            video_info['title'],
            content_map
        )
        
        # Store the conversation app for this video
        if not hasattr(process_video, 'conversation_apps'):
            process_video.conversation_apps = {}
        process_video.conversation_apps[video_id] = conversation_app

        # Generate initial summary and tags
        summary_query = "summarise in 50 words and give tags"
        initial_state = {
            "input": summary_query,
            "chat_history": [],
            "context": "",
            "video_title": video_info['title'],
            "content_map": content_map
        }
        config = {"configurable": {"thread_id": f"video_{video_id}"}}
        
        summary_result = conversation_app.invoke(initial_state, config=config)
        summary_and_tags = summary_result["answer"]

        # Store the summary and tags
        store_summary_and_tags(video_id, summary_and_tags)

        logger.info(f"Video processing completed successfully for video ID: {video_id}")
        return vector_store, retrieval_chain, video_info, content_map, summary_and_tags, []

    except Exception as e:
        logger.error(f"Unexpected error processing video ID {video_id}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None, None, None
    





# def perform_keyword_research(transcript):
#     words = transcript.lower().split()
#     word_count = Counter(words)
#     keywords = word_count.most_common(20)
#     return keywords

# def convert_to_document(item):
#     if isinstance(item, str):
#         return Document(page_content=item)
#     elif hasattr(item, 'page_content'):
#         return item
#     else:
#         return Document(page_content=str(item))
    





from langchain_core.messages import HumanMessage, AIMessage
ChatSession = apps.get_model('app', 'ChatSession')

def process_query(retrieval_chain, query, video_id, video_title, content_map, session_identifier):
    try:
        logger.info(f"Processing query for video {video_id} in session {session_identifier}")
        
        # Get chat history from database for this specific session only
        chat_history = ChatSession.objects.filter(
            session_identifier=session_identifier
        ).order_by('created_at')
        
        # Format the chat history for the LLM
        formatted_history = []
        for chat in chat_history:
            formatted_history.append(HumanMessage(content=chat.question))
            formatted_history.append(AIMessage(content=chat.answer))
            
        # Prepare chain input
        chain_input = {
            "input": query,
            "chat_history": formatted_history,  # This ensures LLM has context from the current session only
            "video_title": video_title,
            "content_map": content_map
        }
        
        # Execute chain and get response
        response = retrieval_chain.invoke(chain_input)
        answer = response.get('answer', "I couldn't generate a response.")
        
        return answer, formatted_history

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
        return f"An error occurred while processing your query: {str(e)}", []
    
def split_summary_content(summary_content):
    parts = summary_content.split("\n\nContent Map:")
    comprehensive_summary = parts[0].replace("Comprehensive Summary:\n", "").strip()
    content_map = parts[1].strip() if len(parts) > 1 else ""
    return comprehensive_summary, content_map
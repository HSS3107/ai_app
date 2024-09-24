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

from pinecone import Pinecone, ServerlessSpec
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from collections import Counter
import tiktoken

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
        logger.error(f"An error occurred while fetching the transcript: {str(e)}")
        return None

def format_original_transcript(transcript, video_info):
    formatted_transcript = f"Title: {video_info['title']}\n"
    formatted_transcript += f"Channel: {video_info['channel']}\n"
    formatted_transcript += f"Published: {video_info['published_at']}\n"
    formatted_transcript += f"Duration: {video_info['duration']}\n"
    formatted_transcript += f"Description: {video_info['description']}\n\n"
    formatted_transcript += "Transcript:\n\n"
    
    for entry in transcript:
        start_time = format_timecode(entry['start'])
        formatted_transcript += f"[{start_time}] {entry['text']}\n"
    
    return formatted_transcript

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

def num_tokens_from_string(string: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def format_transcript_chunk_with_gpt35(transcript_chunk, video_info, is_first_chunk=False, is_last_chunk=False):
    gpt35_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000)
    
    prompt = PromptTemplate.from_template("""
    Format transcript chunk:
    1. {metadata_instruction}
    2. Add speaker labels if possible.
    3. Include brief sentiment for each statement.
    4. {continuation_instruction}

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
            metadatas.append({"video_id": video_id, "chunk_index": i})
        
        vector_store.add_texts(texts, metadatas=metadatas, ids=vector_ids)
        logger.info(f"Added {len(texts)} AI-formatted transcript chunks to vector store for video ID: {video_id}")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating or loading vector store for video_id {video_id}: {str(e)}")
        raise

def setup_llm_chain():
    prompt = PromptTemplate.from_template("""
    You are an AI assistant specializing in analyzing and answering questions about YouTube videos. You're currently focusing on a video titled "{video_title}".
    
    if a question requires data external to the video to be considered, answer using external sources but mention it. 

    Use the following comprehensive summary, content map, and relevant transcript context to provide a detailed and accurate answer:

    Comprehensive Summary:
    {comprehensive_summary}

    Content Map:
    {content_map}

    Relevant Transcript Context:
    {context}

    When answering, please:
    1. Refer to specific parts of the comprehensive summary and content map.
    2. Use exact timestamps from the content map when referencing specific moments in the video.
    3. Mention relevant names when discussing their contributions or statements.
    4. Provide a comprehensive answer that covers all aspects of the question.

    Question: {question}

    Comprehensive Answer:
    """)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    return create_stuff_documents_chain(llm, prompt)

video_summaries = {}

def store_summary(video_id, summary_content, content_map):
    video_summaries[video_id] = {
        'summary': summary_content,
        'content_map': content_map
    }
    logger.info(f"Stored comprehensive summary and content map for video ID {video_id}")
    
def get_summary(video_id):
    summary_data = video_summaries.get(video_id, {
        'summary': "No summary available for this video.",
        'content_map': "No content map available for this video."
    })
    logger.info(f"Retrieved comprehensive summary and content map for video ID {video_id}")
    return summary_data['summary'], summary_data['content_map']

def generate_comprehensive_summary(ai_formatted_transcript, video_info):
    gpt_model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.7)
    
    prompt = PromptTemplate.from_template("""
    Create a comprehensive summary of this video transcript. The summary should:
    1. Start with an introduction mentioning the video title, channel, and main topic.
    2. Include 5-7 main points discussed in the video, with timestamps if available.
    3. Highlight any key quotes or memorable moments.
    4. Identify the overall tone or sentiment of the video.
    5. Conclude with the main takeaway or call to action, if any.
    6. Be around 500-700 words long.

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
    
def generate_content_map(ai_formatted_transcript, video_info):
    gpt_model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.7)
    
    prompt = PromptTemplate.from_template("""
    Create a detailed content map of this video transcript. The content map should:
    1. Divide the video into 5-8 main sections.
    2. For each section, provide:
       a) An exact timestamp range (start-end) using the format [HH:MM:SS-HH:MM:SS]
       b) A title or topic for the section
       c) A brief description of the content covered in that section
    3. Use the exact timestamps provided in the transcript.
    4. Ensure that the sections cover the entire duration of the video.
    5. Include any key quotes or memorable moments within the relevant sections.

    Video: {title}
    Channel: {channel}

    Transcript:
    {ai_formatted_transcript}

    Content Map:
    """)

    try:
        result = gpt_model.invoke(prompt.format(
            title=video_info['title'],
            channel=video_info['channel'],
            ai_formatted_transcript=ai_formatted_transcript
        ))
        return result.content.strip()
    except Exception as e:
        logger.error(f"Error generating content map: {str(e)}")
        return "Error generating content map."

def process_video(video_id):
    logger.info(f"Starting to process video ID: {video_id}")
    try:
        video_info = get_video_info(video_id)
        if not video_info:
            logger.error(f"Failed to get video info for video ID: {video_id}")
            return None, None, None, None, None

        safe_title = sanitize_filename(video_info['title'])
        transcript_path = f"transcripts/{safe_title}.txt"
        ai_formatted_path = f"ai_formatted_transcripts/{safe_title}_ai_formatted.txt"
        summary_path = f"summaries/{safe_title}_summary.txt"
        content_map_path = f"content_maps/{safe_title}_content_map.txt"

        if os.path.exists(transcript_path) and os.path.exists(ai_formatted_path) and os.path.exists(summary_path) and os.path.exists(content_map_path):
            logger.info(f"Transcript, AI-formatted transcript, summary, and content map found for video ID: {video_id}. Loading from files.")
            with open(transcript_path, 'r', encoding='utf-8') as f:
                original_transcript = f.read()
            with open(ai_formatted_path, 'r', encoding='utf-8') as f:
                ai_formatted_transcript = f.read()
            with open(summary_path, 'r', encoding='utf-8') as f:
                comprehensive_summary = f.read()
            with open(content_map_path, 'r', encoding='utf-8') as f:
                content_map = f.read()
        else:
            transcript = get_transcript(video_id)
            if not transcript:
                logger.error(f"Failed to get transcript for video ID: {video_id}")
                return None, None, video_info, None, None

            logger.info(f"Transcript retrieved for video ID: {video_id}. Length: {len(transcript)}")
            original_transcript = format_original_transcript(transcript, video_info)
            os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(original_transcript)

            ai_formatted_transcript = create_ai_formatted_transcript(original_transcript, video_info)
            os.makedirs(os.path.dirname(ai_formatted_path), exist_ok=True)
            with open(ai_formatted_path, 'w', encoding='utf-8') as f:
                f.write(ai_formatted_transcript)

            comprehensive_summary = generate_comprehensive_summary(ai_formatted_transcript, video_info)
            content_map = generate_content_map(ai_formatted_transcript, video_info)

            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(comprehensive_summary)

            os.makedirs(os.path.dirname(content_map_path), exist_ok=True)
            with open(content_map_path, 'w', encoding='utf-8') as f:
                f.write(content_map)

        # Store both summary and content map in memory
        store_summary(video_id, comprehensive_summary, content_map)

        keywords = perform_keyword_research(ai_formatted_transcript)

        # Create vector store using AI-formatted transcript
        try:
            embeddings = create_embeddings()
            index = initialize_pinecone()
            logger.info(f"Embeddings created and Pinecone initialized for video ID: {video_id}")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            ai_formatted_chunks = text_splitter.split_text(ai_formatted_transcript)

            vector_store = create_or_load_vector_store(ai_formatted_chunks, embeddings, index, video_id)
            logger.info(f"Vector store created or loaded for video ID: {video_id} using AI-formatted transcript")
        except Exception as e:
            logger.error(f"Error creating vector store for video ID {video_id}: {str(e)}")
            return None, None, video_info, comprehensive_summary, None

        chain = setup_llm_chain()

        logger.info(f"Video processing completed successfully for video ID: {video_id}")
        return vector_store, chain, video_info, comprehensive_summary, keywords

    except Exception as e:
        logger.error(f"Unexpected error processing video ID {video_id}: {str(e)}")
        return None, None, None, None, None
    
def perform_keyword_research(transcript):
    words = transcript.lower().split()
    word_count = Counter(words)
    keywords = word_count.most_common(20)
    return keywords

def convert_to_document(item):
    if isinstance(item, str):
        return Document(page_content=item)
    elif hasattr(item, 'page_content'):
        return item
    else:
        return Document(page_content=str(item))

def process_query(docsearch, chain, query, video_id, video_title):
    try:
        comprehensive_summary, content_map = get_summary(video_id)
        
        logger.info(f"Using comprehensive summary and content map for video ID {video_id} in query processing")

        docs = docsearch.similarity_search(
            query, 
            filter={"video_id": video_id},
            k=3
        )
        logger.info(f"Retrieved {len(docs)} relevant AI-formatted transcript chunks for context")
        
        documents = [convert_to_document(doc) for doc in docs]

        chain_input = {
            "video_title": video_title,
            "comprehensive_summary": comprehensive_summary,
            "content_map": content_map,
            "context": documents,
            "question": query
        }
        logger.info(f"Chain input for query processing: {chain_input}")

        result = chain.invoke(chain_input)
        
        logger.info(f"Chain output for query: {result}")

        return result['answer'] if isinstance(result, dict) and 'answer' in result else str(result)
    except Exception as e:
        logger.error(f"Error processing query for video_id {video_id}: {str(e)}")
        raise
    
def verify_transcript_and_summary_usage(video_id, query):
    logger.info(f"Verifying AI-formatted transcript and comprehensive summary usage for video ID {video_id}")
    
    # Retrieve AI-formatted transcript and comprehensive summary
    with open(f"ai_formatted_transcripts/{video_id}_ai_formatted.txt", 'r', encoding='utf-8') as f:
        ai_formatted_transcript = f.read()
    comprehensive_summary = get_summary(video_id)
    
    logger.info(f"AI-formatted transcript excerpt: {ai_formatted_transcript[:100]}...")
    logger.info(f"Comprehensive summary excerpt: {comprehensive_summary[:100]}...")
    
    # Simulate query processing
    docsearch = ...  # You would need to retrieve or create this
    chain = setup_llm_chain()
    video_info = get_video_info(video_id)
    
    result = process_query(docsearch, chain, query, video_id, video_info['title'])
    logger.info(f"Query result: {result[:100]}...")

    return "AI-formatted transcript used for context and comprehensive summary used in query processing"

def split_summary_content(summary_content):
    parts = summary_content.split("\n\nContent Map:")
    comprehensive_summary = parts[0].replace("Comprehensive Summary:\n", "").strip()
    content_map = parts[1].strip() if len(parts) > 1 else ""
    return comprehensive_summary, content_map
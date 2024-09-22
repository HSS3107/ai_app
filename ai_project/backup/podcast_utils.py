import os
import hashlib
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore

def load_environment_variables():
    load_dotenv()
    required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY_1', 'PINECONE_API_ENV', 'PINECONE_INDEX_NAME', 'YOUTUBE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

def get_video_info(video_id):
    youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
    request = youtube.videos().list(part='snippet', id=video_id)
    response = request.execute()
    
    if 'items' in response:
        snippet = response['items'][0]['snippet']
        return {
            'title': snippet['title'],
            'channel': snippet['channelTitle'],
            'published_at': snippet['publishedAt']
        }
    return None

def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"An error occurred while fetching the transcript: {str(e)}")
        return None

def format_transcript(transcript, video_info):
    formatted_transcript = f"Title: {video_info['title']}\n"
    formatted_transcript += f"Channel: {video_info['channel']}\n"
    formatted_transcript += f"Published: {video_info['published_at']}\n\n"
    
    current_speaker = None
    for entry in transcript:
        if 'speaker' in entry:
            if entry['speaker'] != current_speaker:
                current_speaker = entry['speaker']
                formatted_transcript += f"\n[{current_speaker}]\n"
        else:
            formatted_transcript += "\n[Speaker]\n"
        
        formatted_transcript += f"{entry['text']}\n"
    
    return formatted_transcript

def create_embeddings():
    return OpenAIEmbeddings()

def initialize_pinecone():
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY_1'))
    index_name = os.getenv('PINECONE_INDEX_NAME')
    
    try:
        index = pc.Index(index_name)
        return index
    except Exception as e:
        print(f"Error connecting to Pinecone index: {str(e)}")
        raise

def create_or_load_vector_store(texts, embeddings, index, video_id):
    vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    
    vector_ids = []
    metadatas = []
    for i, text in enumerate(texts):
        chunk_id = hashlib.md5(f"{video_id}_{i}".encode()).hexdigest()
        vector_ids.append(chunk_id)
        metadatas.append({"video_id": video_id, "chunk_index": i})
    
    vector_store.add_texts(texts, metadatas=metadatas, ids=vector_ids)
    
    return vector_store

def setup_llm_chain():
    prompt = PromptTemplate.from_template("""
    A user has uploaded a YouTube video transcript. The user could be a podcaster/content creator trying to find out what the guest said about a specific topic, 
    a marketer performing keyword analysis or making other marketing-related queries, or a student trying to understand a lecture better.
    
    Below is the context from the transcription of the video. Use this context to answer the question.
    If the user query is unrelated to the video content or the context doesn't provide relevant information, then say: 
    "This doesn't match with the video topic discussed."

    Context: {context}

    Question: {question}
    Answer:
""")
    
    llm = ChatOpenAI(temperature=1.3)
    return LLMChain(llm=llm, prompt=prompt)

def process_query(docsearch, chain, query, video_id):
    docs = docsearch.similarity_search(
        query, 
        filter={"video_id": video_id},
        k=4
    )
    context = "\n".join([doc.page_content for doc in docs])
    result = chain.invoke({"context": context, "question": query})
    return result['text'] if isinstance(result, dict) and 'text' in result else str(result)
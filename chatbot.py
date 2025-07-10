from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader
from groq import Groq
from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from deep_translator import GoogleTranslator
import nest_asyncio
import edge_tts
import asyncio
import os
from langchain.chains import ConversationChain
nest_asyncio.apply()
# 1. Multilingual translation

def make_punctuation_aware(text):
    prompt = f"""
    قم بإضافة علامات الترقيم المناسبة مع تشكيل مناسب زي الفتحة و الكسرة و الضمةو الشدة و التنوين للجملة التالية زي مثلاً التنوين في كلمة أهلاً , هنا انا حطيت تنوين و في كلمة بِك , هنا وضعت كسرة. اكتب الجملة مع علامات الترقيم فقط بدون أي شرح أو مقدمات أو أمثلة أو أرقام أو تعليقات و لا تنسي تشكيل الجملة.:

    {text}
    """
    # Set your Groq API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # You can change to "mixtral-8x7b-32768" or "gemma-7b-it"
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=256
    )
    response = chat_completion.choices[0].message.content.strip()
    return response


def translate(text, source_lang, target_lang):
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except:
        print(f"Translation failed for text")
        return text
def convert_to_egyptian(msa_sentence):
    prompt = f"""
ترجم الجملة التالية من العربية الفصحى إلى اللهجة المصرية العامية فقط. اكتب الترجمة فقط بدون أي شرح أو مقدمات أو أمثلة أو أرقام أو تعليقات و لا تنسي تشكيل الجملة.:

{msa_sentence}
"""
    # Set your Groq API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # You can change to "mixtral-8x7b-32768" or "gemma-7b-it"
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=256
    )
    response = chat_completion.choices[0].message.content.strip()
    return response

def convert_to_orianted(msg_sentence):
    prompt = f"""
ترجم الجملة التالية من اللهجة المصرية إلي العربية الفصحي فقط. اكتب الترجمة فقط بدون أي شرح أو مقدمات أو أمثلة أو أرقام أو تعليقات و لا تنسي تشكيل الجملة.:

{msg_sentence}
"""
    # Set your Groq API Key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    client = Groq(api_key=api_key)
    chat_completion = client.chat.completions.create(
        model="meta-llama/llama-guard-4-12b",  # You can change to "mixtral-8x7b-32768" or "gemma-7b-it"
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=256
    )
    response = chat_completion.choices[0].message.content.strip()
    return response


def create_vector_db(data_dir, persist_directory='./chroma_db'):
    loader = DirectoryLoader(data_dir, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    vector_db.persist()
    print("ChromaDB created and data saved")
    return vector_db
def load_vector_db(persist_directory):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_db
def initialize_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
    return ChatGroq(
        temperature=0.9,
        api_key=api_key, 
        model_name="llama-3.1-8b-instant",
        max_tokens=256
    )
def get_prompt_by_language_adv(language):
    """
    Load a prompt template based on the specified language.
    
    """
    prompts_path = f"prompts/{language}.txt"
    with open(prompts_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    return PromptTemplate(
        input_variables=["context", "question"],
        template=prompt
    )
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
llm = initialize_llm()
# Load or create the vector database

default_lang = "en"
custom_prompt = get_prompt_by_language_adv(default_lang)

chat_chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=custom_prompt
)

def clear_chat_history():
    memory.clear()

    
def chat(user_input , lang=default_lang):
    chat_chain.prompt = get_prompt_by_language_adv(lang)
    return chat_chain.run(user_input)

def print_chat_history():
    for msg in memory.chat_memory.messages:
        print(f"{msg.type.upper()}: {msg.content}")


def tts_edit(text, lang, gender='male', output_path='static/response.mp3'):
    if gender.lower() != 'male':
        sounds = {
            'ar': 'ar-BH-LailaNeural',
            'en': 'en-US-AvaNeural',
            'fr': 'fr-FR-VivienneMultilingualNeural',
            'de': 'de-DE-KatjaNeural'
        }
    else:
        sounds = {
            'ar': 'ar-EG-ShakirNeural',
            'en': 'en-US-ChristopherNeural',
            'fr': 'fr-FR-RemyMultilingualNeural',
            'de': 'de-DE-FlorianMultilingualNeural'
        }
    sound = sounds.get(lang, 'en-US-ChristopherNeural')

    async def run_tts():
        tts = edge_tts.Communicate(
            text=text,
            voice=sound,
            rate="+0%",
            pitch="+2Hz"
        )
        await tts.save(output_path)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(run_tts(), loop)
        future.result()
    else:
        asyncio.run(run_tts())
    return output_path




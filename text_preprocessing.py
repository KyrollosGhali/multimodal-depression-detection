from nltk.tokenize import sent_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
from keras.models import load_model
def open_text_file(file_path):
    """
    Open a text file and read its content.
    Parameters:
        file_path (str): Path to the text file.
    Returns:
        str: Content of the text file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content= file.read()
    return content
def get_dialogue_embedding(text):
    """
    Generate embeddings for the input text using a pre-trained SentenceTransformer model.
    This function tokenizes the text into sentences, computes embeddings for each sentence,
    and combines them using mean and max pooling.
    Parameters:
        text (str): Input text to be embedded.
    Returns:
        np.ndarray: Combined mean and max pooled embeddings of the sentences in the text.
    """
    embedding_model = SentenceTransformer('all-mpnet-base-v2')
    sentences = sent_tokenize(text)
    embeddings = embedding_model.encode(sentences)
    
    mean_pool = np.mean(embeddings, axis=0)
    max_pool = np.max(embeddings, axis=0)
    
    combined = np.concatenate([mean_pool, max_pool])
    return combined
def prediction(text):
    """
    Predict the negative and positive status of the input text using a pre-trained model.
    Parameters:
        text (str): Input text to be analyzed.
    Returns:
        tuple: Negative and positive status probabilities.
    """
    input_data = get_dialogue_embedding(text)
    model_path = r"Models\text_model\text_model.h5"
    text_model = load_model(model_path)
    input_data = np.array([input_data])
    predictions = text_model.predict(input_data)
    negative_status = predictions[0][0]
    positive_status = 1-predictions[0][0]
    return negative_status, positive_status
def main_text(file_path):
    """
    Main function to process the text file and predict its negative and positive status.
    Parameters:
        file_path (str): Path to the text file.
    Returns:
        tuple: Negative and positive status probabilities.
    """
    text = open_text_file(file_path)
    negative_status, positive_status = prediction(text)
    return negative_status, positive_status
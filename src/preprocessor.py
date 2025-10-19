import re
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class RobustTextPreprocessor:
    """
    Text preprocessing for hybrid summarization
    Handles cleaning, tokenization, and normalization
    """
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.stop_words = set()
            self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        try:
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^a-zA-Z\s\.\!\?]', '', text)
            text = text.lower()
            return text.strip()
        except:
            return ""
    
    def preprocess_sentence(self, sentence):
        """Preprocess individual sentence"""
        try:
            cleaned = self.clean_text(sentence)
            if not cleaned:
                return ""
            words = cleaned.split()
            words = [self.lemmatizer.lemmatize(word) for word in words 
                    if word not in self.stop_words and len(word) > 2]
            return ' '.join(words)
        except:
            return ""
    
    def prepare_single_article(self, article, summary=""):
        """Prepare a single article for processing"""
        try:
            original_sentences = sent_tokenize(article)
            preprocessed_sentences = []
            valid_original_sentences = []
            
            for sent in original_sentences:
                processed_sent = self.preprocess_sentence(sent)
                if processed_sent and len(processed_sent.split()) >= 3:
                    preprocessed_sentences.append(processed_sent)
                    valid_original_sentences.append(sent)
            
            if len(valid_original_sentences) >= 3:
                return {
                    'original_sentences': valid_original_sentences,
                    'preprocessed_sentences': preprocessed_sentences,
                    'reference_summary': summary,
                    'num_sentences': len(valid_original_sentences)
                }
            else:
                return None
        except Exception as e:
            print(f"Error processing article: {e}")
            return None

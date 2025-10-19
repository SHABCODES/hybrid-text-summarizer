from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
import networkx as nx
import numpy as np

class ImprovedHybridSummarizer:
    """
    Hybrid summarizer combining K-Means, LDA, and TextRank
    """
    
    def __init__(self, summary_ratio=0.25):
        self.summary_ratio = summary_ratio
        self.vectorizer = TfidfVectorizer(
            max_features=3000, 
            min_df=1, 
            max_df=0.85, 
            ngram_range=(1, 2)
        )
        
    def get_sentence_vectors(self, preprocessed_sentences):
        """Convert sentences to TF-IDF vectors"""
        if len(preprocessed_sentences) < 2:
            return np.ones((len(preprocessed_sentences), 10))
        try:
            vectors = self.vectorizer.fit_transform(preprocessed_sentences).toarray()
            return vectors
        except:
            return np.random.rand(len(preprocessed_sentences), 10)
    
    def perform_kmeans(self, sentence_vectors, n_clusters=None):
        """Cluster sentences using K-Means for diversity"""
        if len(sentence_vectors) < 2:
            return [0] * len(sentence_vectors)
        if n_clusters is None:
            n_clusters = max(2, min(8, int(len(sentence_vectors) ** 0.7)))
        if n_clusters >= len(sentence_vectors):
            n_clusters = max(1, len(sentence_vectors) - 1)
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            cluster_labels = kmeans.fit_predict(sentence_vectors)
            return cluster_labels
        except:
            return [0] * len(sentence_vectors)
    
    def perform_lda(self, preprocessed_sentences, num_topics=4):
        """Perform LDA topic modeling for topic relevance"""
        if len(preprocessed_sentences) < 4:
            return [-1] * len(preprocessed_sentences), None
        try:
            tokenized_sentences = [sent.split() for sent in preprocessed_sentences]
            tokenized_sentences = [tokens for tokens in tokenized_sentences if len(tokens) > 2]
            if len(tokenized_sentences) < 3:
                return [-1] * len(preprocessed_sentences), None
            id2word = corpora.Dictionary(tokenized_sentences)
            corpus = [id2word.doc2bow(tokens) for tokens in tokenized_sentences]
            actual_topics = min(num_topics, len(tokenized_sentences) - 1)
            if actual_topics < 1:
                return [-1] * len(preprocessed_sentences), None
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=id2word,
                num_topics=actual_topics,
                random_state=42,
                passes=15,
                alpha='auto',
                per_word_topics=True
            )
            sentence_topics = []
            for bow in corpus:
                try:
                    topic_scores = lda_model.get_document_topics(bow)
                    if topic_scores:
                        dominant_topic = max(topic_scores, key=lambda x: x[1])
                        if dominant_topic[1] > 0.3:
                            sentence_topics.append(dominant_topic[0])
                        else:
                            sentence_topics.append(-1)
                    else:
                        sentence_topics.append(-1)
                except:
                    sentence_topics.append(-1)
            while len(sentence_topics) < len(preprocessed_sentences):
                sentence_topics.append(-1)
            return sentence_topics, lda_model
        except Exception as e:
            print(f"LDA Error: {e}")
            return [-1] * len(preprocessed_sentences), None
    
    def perform_textrank(self, sentence_vectors):
        """Calculate TextRank scores for sentence importance"""
        if len(sentence_vectors) <= 1:
            return [1.0] * len(sentence_vectors)
        try:
            similarity_matrix = cosine_similarity(sentence_vectors)
            similarity_matrix = np.maximum(similarity_matrix, 0)
            nx_graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(nx_graph, alpha=0.85)
            textrank_scores = [scores[i] for i in range(len(sentence_vectors))]
            return textrank_scores
        except:
            return [1.0] * len(sentence_vectors)
    
    def summarize(self, original_sentences, preprocessed_sentences):
        """Main hybrid summarization function"""
        if len(original_sentences) <= 2:
            return ' '.join(original_sentences[:1])
        try:
            # Get features from all three models
            sentence_vectors = self.get_sentence_vectors(preprocessed_sentences)
            cluster_labels = self.perform_kmeans(sentence_vectors)
            topic_assignments, _ = self.perform_lda(preprocessed_sentences)
            textrank_scores = self.perform_textrank(sentence_vectors)
            
            # Normalize and combine scores
            if textrank_scores and max(textrank_scores) > min(textrank_scores):
                textrank_scores_norm = (textrank_scores - np.min(textrank_scores)) / \
                                     (np.max(textrank_scores) - np.min(textrank_scores))
            else:
                textrank_scores_norm = [0.5] * len(original_sentences)
            
            # Hybrid scoring with position and length bias
            final_scores = []
            for i in range(len(original_sentences)):
                base_score = 0.5 * textrank_scores_norm[i]  # TextRank weight
                position_bias = max(0, 1 - (i / len(original_sentences))) * 0.2
                base_score += position_bias
                if i < len(topic_assignments) and topic_assignments[i] != -1:
                    base_score += 0.3  # Topic relevance bonus
                sentence_len = len(original_sentences[i].split())
                if 8 <= sentence_len <= 25:  # Ideal length bonus
                    base_score += 0.1
                final_scores.append(base_score)
            
            # Strategic sentence selection
            num_to_select = max(1, min(5, int(len(original_sentences) * self.summary_ratio)))
            selected_indices = []
            
            # Pick best from each cluster for diversity
            cluster_best = {}
            for i, cluster_id in enumerate(cluster_labels):
                if cluster_id not in cluster_best or final_scores[i] > final_scores[cluster_best[cluster_id]]:
                    cluster_best[cluster_id] = i
            selected_indices.extend(cluster_best.values())
            
            # Add highest scoring remaining sentences
            remaining_indices = set(range(len(original_sentences))) - set(selected_indices)
            if remaining_indices and len(selected_indices) < num_to_select:
                remaining_list = sorted(remaining_indices, key=lambda x: final_scores[x], reverse=True)
                needed = num_to_select - len(selected_indices)
                selected_indices.extend(remaining_list[:needed])
            
            # Ensure at least one sentence
            if not selected_indices:
                selected_indices = [np.argmax(final_scores)] if final_scores else [0]
            
            # Return summary in original order
            selected_indices.sort()
            summary_sentences = [original_sentences[i] for i in selected_indices]
            return ' '.join(summary_sentences)
            
        except Exception as e:
            print(f"Summarization error: {e}")
            return original_sentences[0] if original_sentences else ""

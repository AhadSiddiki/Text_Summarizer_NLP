import argparse
import torch
import nltk
import spacy
import numpy as np
import networkx as nx
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
# scikit-learn is installed as "scikit-learn" but imported as "sklearn"
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class TextSummarizer:
    def __init__(self, model_name="google/pegasus-large"):
        """
        Initialize the text summarizer with the specified pretrained model.
        
        Args:
            model_name (str): HuggingFace model name for summarization
                Options include:
                - "google/pegasus-large" (default, larger and more creative)
                - "google/pegasus-xsum" (for very concise summaries)
                - "facebook/bart-large-cnn" (original default)
                - "t5-large" (Google T5 large model)
                - "allenai/led-large-16384" (for very long documents)
        """
        print(f"Loading model: {model_name}. First-time initialization may take a few minutes...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Create a summarization pipeline for alternative approach
        self.summarization_pipeline = pipeline(
            "summarization", 
            model=model_name, 
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Initialize spaCy for extractive summarization
        try:
            # Use medium model with word vectors instead of small model
            try:
                self.nlp = spacy.load("en_core_web_md")
            except:
                print("Downloading spaCy medium model with word vectors...")
                spacy.cli.download("en_core_web_md")
                self.nlp = spacy.load("en_core_web_md")
        except:
            print("Falling back to small model. Note: Similarity calculations will be less accurate.")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Downloading spaCy small model...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
        # Initialize rouge scorer for evaluation
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))

    def abstractive_summarize(self, text, max_length=150, min_length=40):
        """
        Generate an abstractive summary using a pre-trained transformer model.
        
        Args:
            text (str): The input text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            
        Returns:
            str: Generated summary
        """
        # Clean the text and truncate if too long
        text = text.replace('\n', ' ').strip()
        
        # Use direct model generation approach
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
        
        # Enable more creative generation with sampling
        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,  
            length_penalty=2.0,  # Higher penalty encourages longer outputs
            min_length=min_length,
            max_length=max_length,
            no_repeat_ngram_size=2,  # Reduced from 3 to allow more flexibility
            early_stopping=True,
            do_sample=True,  # Enable sampling for more creative generation
            top_k=50,        # Consider top 50 tokens at each step
            top_p=0.9,       # Use nucleus sampling
            temperature=0.7  # Temperature > 0.7 increases randomness
        )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Alternative approach using pipeline for texts longer than 1024 tokens
        if len(text.split()) > 800:
            try:
                # Try using the pipeline approach with creative generation
                pipeline_summary = self.summarization_pipeline(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9
                )[0]['summary_text']
                
                # Compare the quality of both summaries
                direct_scores = self.evaluate_summary(text, summary)
                pipeline_scores = self.evaluate_summary(text, pipeline_summary)
                
                # Choose the better summary based on ROUGE-L score
                if pipeline_scores["rougeL"] > direct_scores["rougeL"]:
                    summary = pipeline_summary
            except Exception:
                # Fallback to the direct approach summary
                pass
        
        return summary

    def extractive_summarize_tfidf(self, text, num_sentences=5):
        """
        Generate an extractive summary using enhanced TF-IDF ranking with position importance.
        
        Args:
            text (str): The input text to summarize
            num_sentences (int): Number of sentences to include in the summary
            
        Returns:
            str: Extractive summary consisting of key sentences
        """
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Clean sentences
        clean_sentences = [s.lower() for s in sentences]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(clean_sentences)
        
        # Calculate sentence scores based on TF-IDF values
        sentence_scores = [sum(tfidf_matrix[i].toarray()[0]) for i in range(len(sentences))]
        
        # Position importance - first and last sentences often contain key information
        for i in range(len(sentences)):
            # Boost beginning sentences
            if i < len(sentences) * 0.2:  # First 20% of sentences
                sentence_scores[i] *= 1.25
            # Boost ending sentences
            elif i > len(sentences) * 0.8:  # Last 20% of sentences
                sentence_scores[i] *= 1.15
                
        # Length normalization - avoid extremely short or long sentences
        for i in range(len(sentences)):
            words = len(sentences[i].split())
            if words < 5:  # Too short
                sentence_scores[i] *= 0.7
            elif words > 40:  # Too long
                sentence_scores[i] *= 0.85
        
        # Get the indices of top scoring sentences
        top_indices = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:num_sentences]
        
        # Sort the indices by their original position to maintain the flow of the text
        top_indices.sort()
        
        # Join the top sentences to form the summary
        summary = " ".join([sentences[i] for i in top_indices])
        return summary

    def extractive_summarize_textrank(self, text, num_sentences=5):
        """
        Generate an extractive summary using TextRank algorithm.
        
        Args:
            text (str): The input text to summarize
            num_sentences (int): Number of sentences to include in the summary
            
        Returns:
            str: Extractive summary consisting of key sentences
        """
        # Tokenize the text into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
            
        # Clean sentences and create a clean_text version
        clean_sentences = []
        for s in sentences:
            clean_s = s.lower()
            clean_sentences.append(clean_s)
            
        # Create a sentence similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        # Determine which similarity method to use based on spaCy model
        use_tfidf_fallback = not hasattr(self.nlp, 'has_pipe') or not self.nlp.has_pipe('word2vec')
        
        if use_tfidf_fallback:
            # Use TF-IDF and cosine similarity as a fallback
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(clean_sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        else:
            # Calculate similarity for each sentence pair using spaCy word vectors
            for i in range(len(sentences)):
                for j in range(len(sentences)):
                    if i != j:
                        # Use spaCy for better semantic similarity
                        doc1 = self.nlp(clean_sentences[i])
                        doc2 = self.nlp(clean_sentences[j])
                        
                        # Use spaCy's built-in similarity
                        if doc1.vector_norm and doc2.vector_norm:  # Check if vectors exist
                            similarity_matrix[i][j] = doc1.similarity(doc2)
        
        # Create a graph and add edges
        nx_graph = nx.from_numpy_array(similarity_matrix)
        
        # Calculate PageRank scores
        scores = nx.pagerank(nx_graph)
        
        # Add position bias - reward sentences at the beginning and end
        for i in range(len(sentences)):
            # Boost beginning sentences
            if i < len(sentences) * 0.2:  # First 20% of sentences
                scores[i] *= 1.25
            # Boost ending sentences
            elif i > len(sentences) * 0.8:  # Last 20% of sentences 
                scores[i] *= 1.15
        
        # Get the indices of top scoring sentences
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
        
        # Sort the indices by their original position to maintain the flow of the text
        top_indices.sort()
        
        # Join the top sentences to form the summary
        summary = " ".join([sentences[i] for i in top_indices])
        return summary

    def extractive_summarize(self, text, num_sentences=5, method="textrank"):
        """
        Generate an extractive summary using the specified method.
        
        Args:
            text (str): The input text to summarize
            num_sentences (int): Number of sentences to include in the summary
            method (str): The method to use ('tfidf' or 'textrank')
            
        Returns:
            str: Extractive summary consisting of key sentences
        """
        if method == "tfidf":
            return self.extractive_summarize_tfidf(text, num_sentences)
        else:  # Default to TextRank
            return self.extractive_summarize_textrank(text, num_sentences)

    def hybrid_summarize(self, text, extractive_ratio=0.5, max_length=150, min_length=40, 
                         extractive_method="textrank"):
        """
        Generate a hybrid summary using both extractive and abstractive methods.
        
        Args:
            text (str): The input text to summarize
            extractive_ratio (float): Ratio for extractive summarization (0.0-1.0)
            max_length (int): Maximum length for abstractive summary
            min_length (int): Minimum length for abstractive summary
            extractive_method (str): Method for extractive step ('tfidf' or 'textrank')
            
        Returns:
            str: Hybrid summary
        """
        # First reduce the text with extractive summarization
        extracted_text = self.extractive_summarize(
            text, 
            num_sentences=max(5, int(len(sent_tokenize(text)) * extractive_ratio)),
            method=extractive_method
        )
        
        # For shorter texts, we can sometimes get better results with direct abstractive summarization
        if len(text.split()) < 300:
            direct_summary = self.abstractive_summarize(
                text,
                max_length=max_length,
                min_length=min_length
            )
            
            # Compare quality metrics
            hybrid_summary = self.abstractive_summarize(
                extracted_text,
                max_length=max_length,
                min_length=min_length
            )
            
            # Choose the better summary based on ROUGE-2 score
            hybrid_scores = self.evaluate_summary(text, hybrid_summary)
            direct_scores = self.evaluate_summary(text, direct_summary)
            
            if direct_scores["rouge2"] > hybrid_scores["rouge2"]:
                return direct_summary
            return hybrid_summary
        
        # Then apply abstractive summarization to the extracted text
        abstract_summary = self.abstractive_summarize(
            extracted_text,
            max_length=max_length,
            min_length=min_length
        )
        
        return abstract_summary

    def evaluate_summary(self, original_text, summary):
        """
        Evaluate a summary using ROUGE metrics.
        
        Args:
            original_text (str): The original text
            summary (str): The generated summary
            
        Returns:
            dict: ROUGE scores
        """
        scores = self.scorer.score(original_text, summary)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }
    
    def summarize_from_file(self, file_path, method="hybrid", **kwargs):
        """
        Summarize text from a file.
        
        Args:
            file_path (str): Path to the file containing text
            method (str): Summarization method (extractive, abstractive, or hybrid)
            
        Returns:
            str: Summary of the file content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            if method == "extractive":
                extractive_method = kwargs.get("extractive_method", "textrank")
                return self.extractive_summarize(text, 
                                               num_sentences=kwargs.get("num_sentences", 5),
                                               method=extractive_method)
            elif method == "abstractive":
                return self.abstractive_summarize(text, 
                                               max_length=kwargs.get("max_length", 150),
                                               min_length=kwargs.get("min_length", 40))
            else:  # hybrid
                extractive_method = kwargs.get("extractive_method", "textrank")
                return self.hybrid_summarize(text, 
                                          extractive_ratio=kwargs.get("extractive_ratio", 0.5),
                                          max_length=kwargs.get("max_length", 150),
                                          min_length=kwargs.get("min_length", 40),
                                          extractive_method=extractive_method)
        except Exception as e:
            return f"Error summarizing file: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Text Summarization Tool")
    parser.add_argument("--input", type=str, help="Input text or file path")
    parser.add_argument("--output", type=str, help="Output file path for summary")
    parser.add_argument("--method", type=str, choices=["extractive", "abstractive", "hybrid"], 
                        default="hybrid", help="Summarization method")
    parser.add_argument("--model", type=str, default="google/pegasus-large", 
                        help="Hugging Face model name for summarization. Options: google/pegasus-large, google/pegasus-xsum, facebook/bart-large-cnn, t5-large, allenai/led-large-16384")
    parser.add_argument("--max_length", type=int, default=150, help="Maximum summary length")
    parser.add_argument("--min_length", type=int, default=40, help="Minimum summary length")
    parser.add_argument("--sentences", type=int, default=5,
                        help="Number of sentences for extractive summarization")
    parser.add_argument("--extractive_method", type=str, choices=["tfidf", "textrank"],
                        default="textrank", help="Method for extractive summarization")
    parser.add_argument("--extractive_ratio", type=float, default=0.5,
                        help="Ratio for extractive step in hybrid summarization (0.0-1.0)")
    
    args = parser.parse_args()
    
    summarizer = TextSummarizer(model_name=args.model)
    
    # Determine if input is a file or raw text
    input_text = ""
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            input_text = f.read()
        print(f"Loaded input from file: {args.input}")
    except (FileNotFoundError, TypeError):
        if args.input:
            input_text = args.input
            print("Using provided text input")
        else:
            print("No input provided. Enter text (type 'EOF' on a new line to finish):")
            lines = []
            while True:
                line = input()
                if line.strip() == "EOF":
                    break
                lines.append(line)
            input_text = "\n".join(lines)
    
    # Generate summary based on method
    if args.method == "extractive":
        summary = summarizer.extractive_summarize(
            input_text, 
            num_sentences=args.sentences,
            method=args.extractive_method
        )
    elif args.method == "abstractive":
        summary = summarizer.abstractive_summarize(
            input_text, 
            max_length=args.max_length, 
            min_length=args.min_length
        )
    else:  # hybrid
        summary = summarizer.hybrid_summarize(
            input_text, 
            extractive_ratio=args.extractive_ratio,
            max_length=args.max_length, 
            min_length=args.min_length,
            extractive_method=args.extractive_method
        )
    
    # Output summary
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary saved to {args.output}")
    else:
        print("\nSummary:")
        print("-" * 40)
        print(summary)
        print("-" * 40)
    
    # Print evaluation metrics
    scores = summarizer.evaluate_summary(input_text, summary)
    print("\nEvaluation Metrics:")
    print(f"ROUGE-1: {scores['rouge1']:.4f}")
    print(f"ROUGE-2: {scores['rouge2']:.4f}")
    print(f"ROUGE-L: {scores['rougeL']:.4f}")


if __name__ == "__main__":
    main()

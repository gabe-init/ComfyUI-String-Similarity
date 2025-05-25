import Levenshtein
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from sentence_transformers import SentenceTransformer
import numpy as np

class StringSimilarity:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "actual_text": ("STRING", {"forceInput": True}),
                "ocr_text": ("STRING", {"forceInput": True}),
                "algorithm": (["Levenshtein", "SequenceMatcher", "Jaccard", "Cosine", "WER", "CER", "SentenceTransformer-MPNET", "SentenceTransformer-MiniLM"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "compute_similarity"
    CATEGORY = "Text Analysis"

    def __init__(self):
        self.mpnet_model = None
        self.minilm_model = None

    def compute_similarity(self, actual_text, ocr_text, algorithm):
        actual_text = self.extract_string(actual_text)
        ocr_text = self.extract_string(ocr_text)

        if not actual_text or not ocr_text:
            return ("Error: Both actual_text and ocr_text must be provided and non-empty.",)

        try:
            result = f"Actual Text: '{actual_text}'\nOCR Text: '{ocr_text}'\n\n"

            if algorithm == "Levenshtein":
                distance, similarity = self.levenshtein_similarity(actual_text, ocr_text)
                result += f"Levenshtein Distance: {distance}\nLevenshtein Similarity: {similarity:.2f}"
            elif algorithm == "SequenceMatcher":
                similarity = self.sequence_matcher_similarity(actual_text, ocr_text)
                result += f"SequenceMatcher Similarity: {similarity:.2f}"
            elif algorithm == "Jaccard":
                similarity = self.jaccard_similarity(actual_text, ocr_text)
                result += f"Jaccard Similarity: {similarity:.2f}"
            elif algorithm == "Cosine":
                similarity = self.cosine_similarity_calc(actual_text, ocr_text)
                result += f"Cosine Similarity: {similarity:.2f}"
            elif algorithm == "WER":
                wer = self.word_error_rate(actual_text, ocr_text)
                result += f"Word Error Rate: {wer:.2f}"
            elif algorithm == "CER":
                cer = self.character_error_rate(actual_text, ocr_text)
                result += f"Character Error Rate: {cer:.2f}"
            elif algorithm == "SentenceTransformer-MPNET":
                similarity = self.sentence_transformer_similarity(actual_text, ocr_text, "all-mpnet-base-v2")
                result += f"SentenceTransformer (MPNET) Similarity: {similarity:.2f}"
            elif algorithm == "SentenceTransformer-MiniLM":
                similarity = self.sentence_transformer_similarity(actual_text, ocr_text, "all-MiniLM-L6-v2")
                result += f"SentenceTransformer (MiniLM) Similarity: {similarity:.2f}"
            else:
                return (f"Error: Invalid algorithm '{algorithm}' selected.",)

            return (result,)

        except Exception as e:
            return (f"Error in computation: {str(e)}",)

    @staticmethod
    def extract_string(input_text):
        if isinstance(input_text, (list, tuple)):
            return input_text[0] if input_text else ""
        elif isinstance(input_text, str):
            try:
                parsed = ast.literal_eval(input_text)
                if isinstance(parsed, (list, tuple)):
                    return parsed[0] if parsed else ""
                else:
                    return input_text
            except (ValueError, SyntaxError):
                return input_text
        else:
            return str(input_text)

    @staticmethod
    def levenshtein_similarity(str1, str2):
        str1, str2 = str1.lower(), str2.lower()
        distance = Levenshtein.distance(str1, str2)
        max_length = max(len(str1), len(str2))
        similarity = 1 - (distance / max_length) if max_length > 0 else 1
        return distance, similarity

    @staticmethod
    def sequence_matcher_similarity(str1, str2):
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    @staticmethod
    def jaccard_similarity(str1, str2):
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 1

    @staticmethod
    def cosine_similarity_calc(str1, str2):
        vectorizer = CountVectorizer().fit_transform([str1.lower(), str2.lower()])
        vectors = vectorizer.toarray()
        dot_product = vectors[0].dot(vectors[1])
        norm_product = (vectors[0]**2).sum()**0.5 * (vectors[1]**2).sum()**0.5
        return dot_product / norm_product if norm_product > 0 else 1

    @staticmethod
    def word_error_rate(reference, hypothesis):
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        d = np.zeros((len(ref_words) + 1) * (len(hyp_words) + 1), dtype=np.uint8).reshape((len(ref_words) + 1, len(hyp_words) + 1))
        for i in range(len(ref_words) + 1):
            for j in range(len(hyp_words) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return d[len(ref_words)][len(hyp_words)] / len(ref_words)

    @staticmethod
    def character_error_rate(reference, hypothesis):
        ref_chars = list(reference.lower())
        hyp_chars = list(hypothesis.lower())
        
        d = np.zeros((len(ref_chars) + 1) * (len(hyp_chars) + 1), dtype=np.uint8).reshape((len(ref_chars) + 1, len(hyp_chars) + 1))
        for i in range(len(ref_chars) + 1):
            for j in range(len(hyp_chars) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                if ref_chars[i - 1] == hyp_chars[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)

    def sentence_transformer_similarity(self, str1, str2, model_name):
        if model_name == "all-mpnet-base-v2":
            if self.mpnet_model is None:
                self.mpnet_model = SentenceTransformer(model_name)
            model = self.mpnet_model
        elif model_name == "all-MiniLM-L6-v2":
            if self.minilm_model is None:
                self.minilm_model = SentenceTransformer(model_name)
            model = self.minilm_model
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        embeddings = model.encode([str1, str2])
        return np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))

NODE_CLASS_MAPPINGS = {
    "StringSimilarity": StringSimilarity
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringSimilarity": "String Similarity Node"
}


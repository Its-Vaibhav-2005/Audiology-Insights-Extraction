import os
import re
import json
import spacy
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class AudiologyPipeline:
    def __init__(self, modelDir="./models", templatesDir="./templates", sbertCheckpoint='fine_tuned_audiology_sbert/checkpoint-505'):
        releventStopwords = set(json.load(open(os.path.join(templatesDir, "relevantStopwords.json"), "r"))) 
        self.canonicalSymtoms = json.load(open(os.path.join(templatesDir, "canonicalSymptoms.json"), "r"))
        self.symptomNormalization = json.load(open(os.path.join(templatesDir, "symptomNormalizations.json"), "r"))
        typos = json.load(open(os.path.join(templatesDir, "typos.json"), "r"))

        self.nlp = spacy.load("en_core_web_sm")

        for word in releventStopwords:
            self.nlp.vocab[word].is_stop = False
        
        self.typoMap = {}
        for correct, typoList in typos.items():
            for typo in typoList:
                self.typoMap[typo] = correct

        self.embedder = SentenceTransformer(os.path.join(modelDir, sbertCheckpoint))

        self.clf = joblib.load(os.path.join(modelDir, "audiology_lr_classifier.pkl"))
        self.mlb = joblib.load(os.path.join(modelDir, "audiology_label_binarizer.pkl"))

        self.symptomEmbeddings = self.embedder.encode(self.canonicalSymtoms)

    def _cleanText(self, text):
        # Normalize and clean the input text
        text = text.lower()
        tokens = text.split()
        correctTokens = [self.typoMap.get(token, token) for token in tokens]

        text = " ".join(correctTokens)

        text = re.sub(r'[^a-z\s]', ' ', text)

        sortedNorms = sorted(self.symptomNormalization.items(), key=lambda x: len(x[0]), reverse=True)
        for colloquial, standard in sortedNorms:
            text = re.sub(rf'\b{re.escape(colloquial)}\b', standard, text)

        doc = self.nlp(text)
        cleanedTokens = [token.text for token in doc if not token.is_stop and token.text.strip()]
        return " ".join(cleanedTokens)

    def _extractSymptoms(self, text, theshold=0.55):
        # Extract symptoms from clean text
        extracted = set()
        
        for symptom in self.canonicalSymtoms:
            if symptom.lower() in text:
                extracted.add(symptom)
        
        doc = self.nlp(text)

        phrases = [chunk.text for chunk in doc.noun_chunks if chunk.text.strip()]
        phrases.extend([token.lemma_ for token in doc if token.pos_ in ["ADJ", "VERB", "NOUN"]])
        phrases.append(text)

        if phrases:
            uniquePhrases = list(set(phrases))
            phrasesVector = self.embedder.encode(uniquePhrases)
            similarityMatrix = cosine_similarity(phrasesVector, self.symptomEmbeddings)

            for i, _ in enumerate(uniquePhrases):
                bestIndices = np.argmax(similarityMatrix[i])
                if similarityMatrix[i][bestIndices] >= theshold:
                    extracted.add(self.canonicalSymtoms[bestIndices])
        return list(extracted)

    def _predictConditions(self, text, threshold=0.4):
        # Encodes the text and predicts the highest-probability conditions

        vecInput = self.embedder.encode([text])
        probabilities = self.clf.predict_proba(vecInput)[0]

        validIndices = np.where(probabilities > threshold)[0]
        if len(validIndices)>0:
            sortValidIncdices = validIndices[np.argsort(-probabilities[validIndices])]
            return self.mlb.classes_[sortValidIncdices].tolist()

        return "Unknown Condition (Clinical evaluation recommended)"
    
    def process(self, text):
        # Execute the full pipeline

        cleanText = self._cleanText(text)
        symptoms = self._extractSymptoms(cleanText)
        conditions = self._predictConditions(cleanText)

        return {
            "input_text": text,
            "clean_text": cleanText,
            "extracted_symptoms": symptoms,
            "predicted_conditions": conditions
        }

class Health:
    def __init__(self):
        try:
            self.pipeline = AudiologyPipeline()
        except Exception as e:
            print(f"Error initializing AudiologyPipeline: {e}")
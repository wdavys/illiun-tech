"""Defines ContextRetriever class, which can be used
to find the best (relevant) contexts for a given query"""

import os
from random import sample

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from loader import load_contexts, load_data



class ContextRetriever:
    """Finds the contexts relevant to specific queries"""

    def __init__(self, path="squad1.1/dev-v1.1.json", with_questions=False, recompute=False):
        """Creates a ContextRetriever object to find relevant paragraphs for a query

        Parameters
        ----------
        path : str, optional
            Path to the SQuAD-like dataset, by default "squad1.1/dev-v1.1.json"
        with_questions : bool, optional
            Whether questions should be loaded from the dataset, by default False
        recompute : bool, optional
            Forces computation of embeddings, even if a version exists in cache, by default False
        """

        if with_questions:
            self.contexts, self.questions = load_data(path)
        else:
            self.contexts = load_contexts(path)
            self.questions = None

        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        if (os.path.exists(os.path.join("cache", os.path.basename(path) + ".npy")) and not recompute):
            self.paragraphs_vectors = np.load(os.path.join("cache", os.path.basename(path) + ".npy"))
        else:
            self.paragraphs_vectors = self.model.encode(self.contexts)
            os.makedirs("cache", exist_ok=True)
            np.save(os.path.join("cache", os.path.basename(path)), self.paragraphs_vectors)


    def retrieve_contexts(self, query):
        """Retrieves and ranks the contexts according to their relevance to the query

        Parameters
        ----------
        query : str
            Query to use to rank the contexts

        Returns
        -------
        A tuple with 3 lists:
        - The contexts in order of relevance to the query
        - The corresponding indexes of the contexts in the original list
        - The similarity scores of the contexts with the query (sorted accordingly)
        """

        # Encode the query using the same model used to encode the paragraphs
        question_vector = self.model.encode([query])

        # Search the best contexts using cosine similarity
        scores = cosine_similarity(question_vector, self.paragraphs_vectors).squeeze()
        top_indexes = list(scores.argsort()[::-1])
        
        return ([self.contexts[i] for i in top_indexes], top_indexes, [scores[i] for i in top_indexes])


    def evaluate(self, num_samples=100, repeat=5):
        """Evaluates the model on a random sample of queries from the dataset.
        The model must have been loaded with questions to use this function

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to use for the evaluation, by default 200
        repeat : int, optional
            Number of times to repeat the evaluation (useful to evaluate the variance), by default 1

        Returns
        -------
        A dictionary with two lists (each item in a list corresponds to a run of evaluation):
        - "mean_ranks": the mean rank of the best context for each query
                        (how is the best context ranked in our model)
        - "accuracies": the accuracies of the best contexts for each query
                        (how many times did the model find the best context)
        """

        if self.questions is None:
            raise ValueError("The model must have questions to use this function")

        mean_ranks = []
        accuracies = []
        for _ in range(repeat):
            questions = sample(self.questions, num_samples)
            ranks = []
            accuracies.append(0)
            for question in tqdm(questions):
                _, top_indexes, _ = self.retrieve_contexts(question["question"])
                ranks.append(top_indexes.index(question["context_id"]))
                if top_indexes.index(question["context_id"]) == 0:
                    accuracies[-1] += 1

            accuracies[-1] /= num_samples
            mean_ranks.append(np.mean(ranks))

        return {"mean_ranks": mean_ranks, "accuracies": accuracies}

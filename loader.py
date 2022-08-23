"""Useful routines to load SQuAD-like dataset from json file"""

import json



def load_data(path):
    """Loads contexts and questions from a json file in SQuAD format.

    Parameters
    ----------
    path : str
        Path to the json file containing the dataset

    Returns
    -------
    contexts, questions: tuple, list
        The contexts and questions in the dataset.
        The questions are dictionaries such as {'question': '...', 'context_id': 0}
    """

    with open(path, "rb") as read_file:
        data = json.load(read_file)

    contexts = []
    questions = []
    i = 0
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            contexts.append(paragraph["context"])
            for qa in paragraph["qas"]:
                questions.append({"question": qa["question"], "context_id": i})

            i += 1

    return contexts, questions


def load_contexts(path):
    """Loads contexts from a json file in SQuAD format.

    Parameters
    ----------
    path : str
        Path to the json file containing the dataset

    Returns
    -------
    contexts: list
        The contexts in the dataset.
    """

    with open(path, "rb") as read_file:
        data = json.load(read_file)

    contexts = []
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            contexts.append(paragraph["context"])

    return contexts


def load_questions(path):
    """Loads questions from a json file in SQuAD format.

    Parameters
    ----------
    path : str
        Path to the json file containing the dataset

    Returns
    -------
    questions: list
        The questions in the dataset. The questions are dictionaries
        such as {'question': '...', 'context_id': 0}
    """

    with open(path, "rb") as read_file:
        data = json.load(read_file)

    questions = []
    i = 0
    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                questions.append({"question": qa["question"], "context_id": i})
            i += 1

    return questions

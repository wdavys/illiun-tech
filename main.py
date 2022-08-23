from argparse import ArgumentParser
from relevant_contexts import ContextRetriever



parser = ArgumentParser(description="Finds the best context for a given query.")
parser.add_argument("query", help="Query to use to find the best context")

parser.add_argument(
    "-n",
    "--n_contexts",
    dest="n_contexts",
    help="Number of relevant contexts to output.")

parser.add_argument(
    "-d",
    "--dataset",
    dest="path",
    help="Path to the dataset to use. Must be a json file in SQuAD format. Default is the SQuAD v1.1 dev dataset."
)

parser.add_argument(
    "-f",
    "--force",
    dest="recompute",
    type=bool,
    help="Forces computation of paragraphs embeddings, even if a cache version is available."
)

parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    type=bool,
    help="Evaluates the performance of the task."
)

args = parser.parse_args()
if args.path is None:
    args.path = "squad1.1/dev-v1.1.json"

if args.recompute is None:
    args.recompute = False

if args.n_contexts is None:
    args.n_contexts = 5
    
if args.evaluate is None:
    args.evaluate = False

if args.evaluate:
    finder = ContextRetriever(path=args.path, recompute=args.recompute, with_questions=True)
    print(finder.evaluate())
else:
    finder = ContextRetriever(path=args.path, recompute=args.recompute)
    contexts, _, scores = finder.retrieve_contexts(args.query)

    for i in range(int(args.n_contexts)):
        print("\n-------------------")
        print(f"Context n°{i+1}:")
        print("-------------------")
        print(contexts[i])
        print(f"Similarity score: {scores[i]:.2%}")

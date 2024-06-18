import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combines a csv file by combining a certain number of sentences")
    parser.add_argument('num_sentences', type=int, help="Number of sentences to combine.")
    args = parser.parse_args()

main(args.num_sentences)

df = pd.read_csv('PrelimTranscript_sentences.csv')

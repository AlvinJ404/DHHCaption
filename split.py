import nltk
import csv
import argparse

# Download the punkt tokenizer if not already downloaded
nltk.download('punkt')

# Function to read the file and split it into sentences
def split_text_into_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    sentences = nltk.sent_tokenize(text)
    return sentences

# Function to create a dictionary with combined sentences
def create_combined_sentence_dict(sentences, num_sentences):
    combined_sentences = [
        ' '.join(sentences[i:i + num_sentences])
        for i in range(0, len(sentences), num_sentences)
    ]
    sentence_dict = {i + 1: sentence for i, sentence in enumerate(combined_sentences)}
    return sentence_dict

# Function to save the dictionary to a CSV file
def save_dict_to_csv(dictionary, csv_file_path):
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Sentence Number', 'Sentence'])
        
        for key, value in dictionary.items():
            writer.writerow([key, value])

def main(text_file_path, num_sentences):
    sentences = split_text_into_sentences(text_file_path)
    sentence_dict = create_combined_sentence_dict(sentences, num_sentences)
    csv_file_path = f'sentences_combined_{num_sentences}.csv'
    save_dict_to_csv(sentence_dict, csv_file_path)
    print(f"Sentences saved to {csv_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine sentences from a text file and save them into a CSV file.")
    parser.add_argument('num_sentences', type=int, help="Number of sentences to combine into each entry.")
    parser.add_argument('text_file', type=str, help="Path to the input text file.")

    args = parser.parse_args()

    main(args.text_file, args.num_sentences)

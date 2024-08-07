from openai import OpenAI
import pandas as pd
import os
import time

FRE_definition = "The Flesch Reading Ease (FRE) scale is calculated by the following: FRE = 206.835 - 1.015 * (totalWords / totalSentences) - 84.6 * (totalSyllables / totalWords). It assigns higher scores to texts that are easier to read."
instruction = "For this sentence, summarize with FRE in ten intervals (within the range of 10-20, 20-30, up to 90-100). During the summary process, paraphrase the original sentence, adhere to the original sentence, and do not add any additional semantic meaning to the original sentence. Each summarization of a sentence is referred to as a summary. Your summary's actual FRE needs to align with the desired range of FRE. This is the most important step. If they do not align, adjust the summary again."
output = "Output your result in the following format: 'Original sentence: XXX' [on a new line] 'Summary: XXX' [on a new line] 'Level: X'. Level is the expected FRE (Level 1 for expected FRE=10-20, Level 2 for expected FRE=20-30, ... Level 9 for expected FRE = 90-100). If you do not follow this format (for example, add new symbols), the result will be considered invalid. There are nine summaries, and hence nine rows."

client = OpenAI(api_key=os.environ.get('OPENAI_DHH_KEY'))

# Credit to Richie Cotton's guide on querying the GPT API, found at:
# https://www.datacamp.com/tutorial/using-gpt-models-via-the-openai-api-in-python
def predict(text):
    completion = client.chat.completions.create(
        model="gpt-4o", 
        messages=[
            {"role": "system", "content": FRE_definition + ' ' + instruction + ' ' + output},
            {"role": "user", "content": text},
        ]
    )
    return completion

def main():
    start_time = time.time()
    with open('by_word_original/50_word_original.txt', 'r') as f: # original text file here
        sentences = f.readlines()
    
    with open('by_word_summary/50_word_summary/50_word_summary.txt', 'a', encoding='utf-8') as file: # summary text file here
        iteration = 1
        for sentence in sentences:
            completion = predict(sentence).choices[0].message.content
            print(completion)
            iteration += 1
            print('\n' + 'iteration: ' + str(iteration) + '\n')
            file.write(completion + '\n')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
import os
import sys
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.probability import FreqDist
from termcolor import colored
from instructions import display_instructions
import matplotlib.pyplot as plt
import seaborn as sns

# Declare text as a global variable
text = ""

# Function to display word frequency distribution histogram
def display_word_frequency_histogram(freq_dist, output_dir):
    plt.figure(figsize=(10, 6))
    freq_dist.plot(30, cumulative=False)
    plt.title('Word Frequency Distribution')
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'word_frequency_histogram.png'))
    plt.close()

# Function to display named entity recognition pie chart
def display_named_entity_pie_chart(named_entities, output_dir):
    entity_counts = {}
    for entity in named_entities:
        if isinstance(entity, nltk.Tree):
            entity_label = ' '.join([word for word, tag in entity.leaves()])
            entity_counts[entity_label] = entity_counts.get(entity_label, 0) + 1

    plt.figure(figsize=(8, 8))
    plt.pie(entity_counts.values(), labels=entity_counts.keys(), autopct='%1.1f%%', startangle=140)
    plt.title('Named Entity Recognition')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'named_entity_pie_chart.png'))
    plt.close()

# Function to process text
def process_text():
    global text  # Access the global text variable
    words, sentences = tokenize_text(text)
    tagged_tokens = tag_part_of_speech(words)
    named_entities = recognize_named_entities(tagged_tokens)
    freq_dist = analyze_frequency(words)
    return words, sentences, tagged_tokens, named_entities, freq_dist

# Function to generate analysis report
def generate_report(words, sentences, tagged_tokens, named_entities, freq_dist, output_dir):
    report = "Text Analysis Report\n\n"
    report += f"Number of sentences: {len(sentences)}\n"
    report += f"Number of words: {len(words)}\n\n"
    report += f"Word Frequency:\n{freq_dist.most_common(10)}\n\n"
    save_report(report, os.path.join(output_dir, 'analysis_report.txt'))

    display_word_frequency_histogram(freq_dist, output_dir)
    display_named_entity_pie_chart(named_entities, output_dir)

# Function to tokenize text
def tokenize_text(text):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    return words, sentences

# Function to tag part of speech
def tag_part_of_speech(words):
    tagged_tokens = [pos_tag(sentence) for sentence in words]
    return tagged_tokens

# Function to recognize named entities
def recognize_named_entities(tagged_tokens):
    named_entities = [ne_chunk(tagged_sentence) for tagged_sentence in tagged_tokens]
    return named_entities

# Function to analyze word frequency
def analyze_frequency(words):
    all_words = [word for sentence in words for word in sentence]
    freq_dist = FreqDist(all_words)
    return freq_dist

# Function to save report
def save_report(report, filename):
    with open(filename, 'w') as file:
        file.write(report)

# Main function
def main():
    global text  # Access the global text variable
    print(colored("Welcome to the Text Process and Analysis Tool", "blue"))
    
    while True:
        print(colored("Main Menu:", "yellow"))
        print("1) Instructions")
        print("2) Input File")
        print("3) Process")
        print("4) More Options")
        print("5) Exit")

        choice = input(colored("Please select an option: ", "blue"))

        if choice == "1":
            display_instructions()
        elif choice == "2":
            input_file_path = input(colored("Enter the path to the input file: ", "yellow"))
            try:
                with open(input_file_path, 'r') as file:
                    text = file.read()
                print(colored("Input file loaded successfully.\n", "green"))
            except FileNotFoundError:
                print(colored("File not found. Please enter a valid file path.\n", "red"))
        elif choice == "3":
            if not text:  # Check if text is empty
                print(colored("Please load an input file before processing.\n", "red"))
                continue
            output_dir = 'analysis_results'
            os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
            tokens, sentences, tagged_tokens, named_entities, freq_dist = process_text()
            generate_report(tokens, sentences, tagged_tokens, named_entities, freq_dist, output_dir)
            print(colored(f"Analysis report and graphs saved to '{output_dir}' directory.\n", "green"))
        elif choice == "4":
            print("More options coming soon!")
        elif choice == "5":
            print(colored("Exiting...", "green"))
            sys.exit()
        else:
            print(colored("Invalid choice. Please select a valid option.\n", "red"))

if __name__ == "__main__":
    main()

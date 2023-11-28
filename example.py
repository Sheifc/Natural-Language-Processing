### Text Processing and Structured Data Conversion Example

'''Importing libraries:

In the first line, we import the nltk library, which is a tool for natural language 
processing.
In the second line, we import specific modules from nltk that we will need, such as 
stopwords (for irrelevant words) and word_tokenize (for splitting the text into words).
In the third line, we import the spacy library, which is another tool for more advanced 
natural language processing.'''

# step 1: Importing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

'''Text preprocessing:

We download necessary resources for preprocessing, such as the list of stopwords and the 
tokenization model.
We define an example text that we want to process.
We convert the text to lowercase and tokenize it into words.
We filter out the stopwords (like "is", "an", "for") and keep only the relevant words.'''

# Step 2: Text preprocessing
nltk.download('stopwords')
nltk.download('punkt')

text = "This is an example sentence for text preprocessing."
stop_words = set(stopwords.words('english'))

tokens = word_tokenize(text.lower())  # Tokenization
filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]  # Filtering out irrelevant words

'''Entity extraction:

We load the spaCy natural language processing model for the English language.
We process the text with spaCy to obtain a document object (doc).
We extract the named entities from the document, which are elements like person names, 
locations, organizations, etc.'''

# Step 3: Entity extraction
nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

entities = [(entity.text, entity.label_) for entity in doc.ents]  # Named entity extraction

'''Text analysis:

We extract the noun phrases from the document, which are groups of words 
that act as nouns.'''

# Step 4: Text analysis (example of topic analysis using spaCy)
noun_phrases = [chunk.text for chunk in doc.noun_chunks]  # Extraction of nominal sentences

'''Printing results:

We print the structured_data dictionary to the console to see the obtained results.'''

# Step 5: Conversion to structured data (example of creating a structured information dictionary)
structured_data = {
    'text': text,
    'filtered_tokens': filtered_tokens,
    'entities': entities,
    'noun_phrases': noun_phrases
}

print(structured_data)

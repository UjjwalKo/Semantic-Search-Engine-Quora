# Semantic Search

This project demonstrates a semantic search application using the LaBSE model from the `sentence-transformers` library. The code performs semantic searches on a dataset of Quora question titles, allowing for the retrieval of similar questions based on meaning, rather than exact keyword matching.

## Requirements

Install the necessary Python libraries using the following command:

```terminal
pip install transformers sentence-transformers torch pandas
```

## Project Setup

### 1. Mount Google Drive
We are using Google Colab to run this code and access the dataset stored in Google Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Load Dataset
The dataset (`quora_titles.csv`) contains Quora question titles, which will be used for the semantic search.

```python
import pandas as pd

# Load dataset from Google Drive
df = pd.read_csv('/content/drive/MyDrive/Semantic/quora_titles.csv')

# Display first few rows
df.head(3)

# Check shape of the dataset
df.shape

# Check for missing values
df.isnull().sum()

# Drop missing values
df.dropna(inplace=True)

# Extract titles into a list
titles = df['Titles'].to_list()
```

### 3. Load Pre-trained Model
We are using the LaBSE (Language-Agnostic BERT Sentence Embedding) model to encode the question titles.

```python
from sentence_transformers import SentenceTransformer

# Load LaBSE model
model = SentenceTransformer('LaBSE')
```

### 4. Encode Titles
We encode the titles into embeddings, which will be used for semantic similarity searches.

```python
# Encode the first 15,000 titles
embed = model.encode(titles[:15000], show_progress_bar=True, convert_to_tensor=True)
```

### 5. Define Search Function
A search function is defined to encode the input question and find the most similar titles in the dataset using cosine similarity.

```python
import time
from sentence_transformers import util

def search(inp_question):
    start_time = time.time()
    question_embedding = model.encode(inp_question, convert_to_tensor=True)
    
    hits = util.semantic_search(question_embedding, embed)
    end_time = time.time()
    hits = hits[0]
    
    print("Input question:", inp_question)
    print("Results (after {:.3f} seconds):".format(end_time - start_time))
    for hit in hits[0:2]:
        print("\t{:.3f}\t{}".format(hit['score'], titles[hit['corpus_id']]))

# Example usage
search("Men and Women")
```

## How It Works
- **Dataset**: The Quora dataset consists of question titles, which are preprocessed to remove missing values.
- **Model**: We use the LaBSE model to convert the question titles into embeddings.
- **Semantic Search**: For a given input question, we compute its embedding and use cosine similarity to find the closest match from the encoded titles.

## Example
When you search for the question "Men and Women," the function will return the closest matching Quora title based on semantic meaning.

```python
search("Men and Women")
```

---

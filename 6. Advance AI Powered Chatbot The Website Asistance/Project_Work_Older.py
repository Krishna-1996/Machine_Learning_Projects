
# %%
# 📍 Phase 1: Planning & Scope Definition

# Step 1: Check Permission: https://data.gov/robots.txt

# Step 2: Explore the Sitemap https://data.gov/sitemap.xml

# %%
# 📍 Phase 2: Data Acquisition

# already completed

# %%
#📍 Phase 3: 📄Web Scraping aka Data Preprocessing
import pandas as pd

# Load the CSV file
df = pd.read_csv('scraped_full_pages_updated.csv')

# Preview
display(df.head(6))# Chucking the data

# 3.1 Content Cleaning
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Fix encoding
    text = text.encode('utf-8', 'ignore').decode('utf-8')

    # Remove boilerplate repeated across pages
    boilerplate_patterns = [
        r"An official website of the United States government.*?site is secure\.",
        r"Federal government websites often end in \.gov or \.mil.*?site is secure\."
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    # Replace fancy quotes
    text = re.sub(r'[“”]', '"', text)
    text = re.sub(r"[‘’]", "'", text)

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

df['clean_text'] = df['text'].apply(clean_text)

# 3.2 Chunking
import nltk
# Download the missing resource
nltk.download('punkt_tab')
nltk.download('punkt') # Ensure 'punkt' is also downloaded

from nltk.tokenize import sent_tokenize


def chunk_text(text, max_tokens=500):
    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current.split()) + len(sentence.split()) <= max_tokens:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks

# Expand into rows
chunk_data = []
for idx, row in df.iterrows():
    chunks = chunk_text(row['clean_text'])
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            'chunk_id': f"{idx}_{i}",
            'url': row['url'],
            'title': row['title'],
            'text': chunk
        })

# New DataFrame
chunks_df = pd.DataFrame(chunk_data)
chunks_df.head()


chunks_df.to_csv('cleaned_chunks.csv', index=False)
chunks_df.head()
print('Cleaned Chunks file is saved')

# %%

# 📍Phase 4: Embedding + Vector Store with FAISS in Colab step-by-step.
# from google.colab import drive
# drive.mount('/content/drive')
# Step 1: Setup environment & install needed libs


# Step 2: Load your chunked data (example assumes CSV with 'chunk_id', 'url', 'title', 'text')
import pandas as pd

df = pd.read_csv('cleaned_chunks.csv')
print(f"Loaded {len(df)} chunks.")
df.head()

# Step 3: Load the embedding model & encode chunks
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode chunks - this returns a numpy array of embeddings
embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
print(f"Encoded {len(embeddings)} chunks into embeddings with shape {embeddings.shape}")

# Step 4: Step 4: Build the FAISS index
import faiss
import numpy as np

embedding_dim = embeddings.shape[1]

# Initialize FAISS index
index = faiss.IndexFlatL2(embedding_dim)  # L2 distance

# Add embeddings to index
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors.")

# Step 5: Save the index and metadata file in drive folder for future use.
import os
# old code: os.makedirs(drive_path, exist_ok=True) 
save_folder = 'faiss_data'
os.makedirs(save_folder, exist_ok=True)  # creates the folder if not present
faiss.write_index(index, f"faiss_index.bin")
df.to_csv(f"metadata.csv", index=False)

print("FAISS index and metadata saved successfully.")

# Step 6: Loading later for querying
# Load FAISS index
index = faiss.read_index(f"faiss_index.bin")

# Load metadata
metadata = pd.read_csv(f"metadata.csv")

# Step 7
query = "What is the purpose of the Open Data for Agriculture initiative?"
query_embedding = model.encode([query])

k = 5  # number of nearest neighbors to return
distances, indices = index.search(query_embedding, k)

for i, idx in enumerate(indices[0]):
    print(f"Result {i+1}:")
    print("URL:", metadata.loc[idx, 'url'])
    print("Title:", metadata.loc[idx, 'title'])
    print("Text snippet:", metadata.loc[idx, 'text'][:300])
    print(f"Distance: {distances[0][i]:.4f}")
    print('---')



# %%
# Create a model and save it in google drive for future testing/use.

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
 
model_name = "mosaicml/mpt-7b"
model_save_path = "./models/mosaicml/mpt-7b"
 
# Create the directory if it doesn't exist
os.makedirs(model_save_path, exist_ok=True)
 
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
 
# Save tokenizer and model locally
tokenizer.save_pretrained(model_save_path)
model.save_pretrained(model_save_path)
 
print(f"Model and tokenizer saved to {model_save_path}")



# %%
# 🎯 Phase 5: Small Language Model (SLM) Inference using OpenChat 3.5‑7B

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "openchat/openchat-3.5-0106"  # You can pick 3.5-1210 or 3.5-16k variants
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Optional: Set up a chat-style pipeline
chat = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    eos_token_id=tokenizer.eos_token_id,
)

# 🔁 Use top 2 FAISS chunks as context
top_k_texts = [metadata.loc[idx, 'text'] for idx in indices[0][:2]]
question = "How is open data used in agriculture?"
context = "\n".join(top_k_texts)

prompt = f"""You are an agricultural data expert. Based only on the context below, summarize how open data is used in agriculture in a clear, structured way using bullet points.

Context:
{context}

Question: {question}
Answer:"""

answer = chat(prompt)[0]['generated_text']
print(answer)


# ✂️ Truncate context if total prompt exceeds 512 tokens
def truncate_prompt(tokenizer, context_chunks, base_template, question, max_tokens=512):
    for k in range(len(context_chunks), 0, -1):
        partial_context = "\n".join(context_chunks[:k])
        prompt = base_template.format(context=partial_context, question=question)
        tokenized = tokenizer(prompt, return_tensors="pt", truncation=False)
        if tokenized["input_ids"].shape[1] <= max_tokens:
            return prompt
    # Fallback to minimal prompt
    return base_template.format(context="", question=question)

# ✨ Build safe prompt
safe_prompt = truncate_prompt(
    tokenizer=tokenizer,
    context_chunks=top_k_texts,
    base_template=base_prompt_template,
    question=question,
    max_tokens=512
)

# Tokenize prompt, truncate input tokens to max 512 tokens explicitly:
inputs = tokenizer(
    safe_prompt,
    max_length=512,      # input max tokens
    truncation=True,     # truncate input tokens if longer than max_length
    return_tensors="pt"
)

# Run the model with sampling parameters to get varied outputs
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=50,
    do_sample=True,          # Enable sampling (randomness) for diversity
    temperature=0.7,         # Creativity level: 0.7 is moderately creative
    top_k=50,                # Limits next token selection to top 50 choices
    top_p=0.9,               # Nucleus sampling: consider tokens until 90% probability mass
    num_return_sequences=1   # Generate 1 different answers(change as per req)
)

# Decode and print all generated outputs
for i, output in enumerate(outputs):
    print(f"🧠 Answer {i+1}:\n", tokenizer.decode(output, skip_special_tokens=True), "\n---")



 # %%

# 📍 Phase 6: RAG System Integration
# Step 1: Embed the user query with the same embedding model (sentence-transformers)
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from urllib.parse import urlparse


# Load model (reuse your embedding model)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Example user query
user_query = "How is open data used in agriculture?"

# Embed query
query_embedding = embedding_model.encode([user_query])


# Step 2: Search your FAISS index for top relevant chunks
# Load your FAISS index and metadata if not already loaded
index = faiss.read_index(f"faiss_index.bin")
metadata = pd.read_csv(f"metadata.csv")
# Helper: Extract topic from URL path (e.g., 'food', 'climate')
def extract_topic_from_url(url):
    path = urlparse(url).path  # e.g. /food/open-ag-data/
    parts = path.strip('/').split('/')
    if parts:
        return parts[0]
    return "unknown"

# Prompt template with topic/title context hint
base_prompt_template = """
You are an expert assistant specialized in agriculture, climate, energy, and ocean data.

Use ONLY the context below to answer the question factually and in detail. If you find relevant information across multiple chunks, combine it into a cohesive answer.

Each chunk is tagged with its source topic and title.

If the context does not have the answer, say: "I don’t know based on the provided data."

------------------------
Context:
{context}
------------------------

Question: {question}

Answer:
"""



def rag_chat_loop(
    embedding_model,
    faiss_index,
    metadata_df,
    tokenizer,
    lm_model,
    base_prompt_template,
    max_input_tokens=512,
    top_k=5,
    max_new_tokens=300,
):
    import torch

    def truncate_prompt(tokenizer, context_chunks, base_template, question, max_tokens=512):
        # Attempt to fit as many chunks as possible within token limit
        for k in range(len(context_chunks), 0, -1):
            partial_context = "\n".join(context_chunks[:k])
            prompt = base_template.format(context=partial_context, question=question)
            tokenized = tokenizer(prompt, return_tensors="pt", truncation=False)
            if tokenized["input_ids"].shape[1] <= max_tokens:
                return prompt
        # If no chunks fit, return prompt with empty context
        return base_template.format(context="", question=question)

    print("✅ RAG Chatbot Ready!")
    print("Type your question. Type 'exit' or 'quit' to stop.\n")

    while True:
        user_query = input("You: ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Step 1: Embed user query
        query_embedding = embedding_model.encode([user_query])

        # Step 2: Search FAISS index for relevant chunks
        distances, indices = faiss_index.search(query_embedding, top_k)

        # Step 3: Build context chunks with topic and title info prepended
        top_k_texts = []
        for idx in indices[0]:
            if idx == -1:
                continue  # skip if invalid index
            row = metadata_df.loc[idx]
            topic = extract_topic_from_url(row['url'])
            title = row.get('title', 'No Title').strip()
            text = row.get('text', '').strip()

            # Format chunk with topic and title tags
            chunk = f"[Topic: {topic}] [Title: {title}]\n{text}"
            top_k_texts.append(chunk)

        # Debug print retrieved chunks (optional)
        print("\n🔍 Retrieved Context Chunks:")
        for i, chunk in enumerate(top_k_texts):
            print(f"\nChunk {i+1}:\n{chunk[:500]}...\n")  # preview first 500 chars

        # Step 4: Prepare prompt (truncate if too long)
        prompt = truncate_prompt(tokenizer, top_k_texts, base_prompt_template, user_query, max_tokens=max_input_tokens)

        # Step 5: Tokenize and generate answer
        inputs = tokenizer(prompt, max_length=max_input_tokens, truncation=True, return_tensors="pt")

        outputs = lm_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding or beam search might improve factuality
            num_beams=3,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Bot: {answer}\n")




# %%
# Run the Chatbot in Loop:
rag_chat_loop(
    embedding_model=embedding_model,
    faiss_index=index,
    metadata_df=metadata,
    tokenizer=tokenizer,
    lm_model=model,
    base_prompt_template=base_prompt_template,
    max_input_tokens=512,
    top_k=5,
    max_new_tokens=300
)


# %%

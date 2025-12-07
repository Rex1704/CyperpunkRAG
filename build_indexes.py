import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# --- Configuration ---
MODEL_NAME = 'all-mpnet-base-v2'  # Excellent balance of speed and quality
FILES = {
    "lore": "data/embeddings_ready/cyberpunk_embedding_documents.json",
    "timeline": "data/embeddings_ready/cyberpunk_timeline_embedding_documents.json",

}


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_and_save_index(name, data_dict, model):
    print(f"\nProcessing {name} index...")

    # 1. Prepare Data
    # We need two lists: IDs (keys) and Documents (values)
    ids = list(data_dict.keys())
    documents = list(data_dict.values())

    print(f"   - Encoding {len(documents)} documents (this may take a moment)...")

    # 2. Generate Embeddings
    # This converts text into a list of 768-dimensional vectors
    embeddings = model.encode(documents, show_progress_bar=True)

    # 3. Create FAISS Index
    # Convert to float32 (required by FAISS)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]  # Should be 768 for mpnet

    # IndexFlatL2 measures 'Euclidean distance' (closer is more similar)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # 4. Save Index and Metadata
    # We save the FAISS index (the math) separate from the text (the content)
    faiss.write_index(index, f"cyberpunk_{name}.index")

    # We need to save the 'ids' and 'documents' so we can look them up later
    # by the index number FAISS returns (0, 1, 2...)
    metadata = {
        "ids": ids,
        "documents": documents
    }
    with open(f"cyberpunk_{name}_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"{name} index saved!")


# --- Main Execution ---
def main():
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    for name, filepath in FILES.items():
        try:
            data = load_json(filepath)
            build_and_save_index(name, data, model)
        except FileNotFoundError:
            print(f"Skipped {name}: File not found ({filepath})")

    print("\nAll indexes built successfully!")


if __name__ == "__main__":
    main()
    # Done in colab
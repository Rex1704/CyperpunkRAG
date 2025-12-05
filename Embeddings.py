import json
import re
import unicodedata
import spacy

INPUT_FILE = "data/cyberpunk_fully_normalized_corpus.json"
OUTPUT_FILE = "data/embeddings_ready/cyberpunk_embedding_documents.json"
NER_OUTPUT_FILE = "data/indexes/cyberpunk_ner_entities.json"

TIMELINE_INPUT_FILE = "data/cyberpunk_timeline.json"
TIMELINE_OUTPUT_FILE = "data/embeddings_ready/cyberpunk_timeline_embedding_documents.json"

KEYS_TO_SKIP = ['template_type']


MAPPING_OUTPUT_FILE = "data/indexes/document_type_mapping.json"

# Keywords used to identify the document type
CLASSIFICATION_RULES = {
    'character': ['character', 'person', 'figure'],
    'location': ['location', 'place', 'city', 'district', 'area', 'building'],
    'concept': ['concept', 'technology', 'gear', 'item', 'weapon', 'cyberware', 'skill', 'ability'],
    'organization': ['corporation', 'corp', 'organization', 'gang', 'group', 'company']
}


def get_document_type(article_data):
    """Classifies an article based on keywords found in its text/metadata."""
    # Concatenate all text/values into a single string for searching
    full_text = ""
    for key, value in article_data.items():
        if isinstance(value, str):
            full_text += value.lower() + " "
        elif isinstance(value, list):
            full_text += " ".join(value).lower() + " "

    # Check for Category tags specifically
    category_match = re.search(r'category:([^\s]+)', full_text)
    if category_match:
        category_tag = category_match.group(1).lower()
    else:
        category_tag = ""

    # Check against classification rules
    for doc_type, keywords in CLASSIFICATION_RULES.items():
        # Check if any keyword matches the text OR the category tag
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', full_text) or keyword in category_tag:
                return doc_type

    # Default type if no match is found
    return 'concept'

# Load the large spaCy model (must be downloaded first)
print("Loading spaCy model (en_core_web_lg)...")
try:
    # Use 'enable' to only run the NER pipeline component, speeding up the process
    # We are excluding the components that are not needed for simple entity extraction
    nlp = spacy.load("en_core_web_lg", exclude=['tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading spaCy model. Did you run 'python -m spacy download en_core_web_lg'? {e}")
    exit()


def extract_entities(text):
    """Processes text and returns a list of unique, cleaned entities."""
    if not text:
        return []

    doc = nlp(text)
    entities = {}  # Use a dictionary to ensure uniqueness

    # Focus on the most relevant entity types for Cyberpunk lore
    # GPE/LOC (Location), ORG (Organization/Corporation), PERSON (Character), PRODUCT/WORK_OF_ART (Items/Shards)
    relevant_labels = ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'WORK_OF_ART']

    for ent in doc.ents:
        if ent.label_ in relevant_labels:
            # Clean and normalize the entity text
            entity_text = ent.text.strip().replace('\n', ' ')
            entity_text = re.sub(r'\s+', ' ', entity_text)  # Normalize whitespace

            # Use lowercased text as the key to ensure uniqueness
            key = entity_text.lower()

            # Store the entity's normalized text and label
            entities[key] = {
                'text': entity_text,
                'label': ent.label_
            }

    # Return a list of the unique entity objects
    return list(entities.values())

def normalize_text(text):
    """
    Applies the full cleaning pipeline for embedding readiness:
    lowercasing, punctuation cleanup, normalization (ō -> o), and whitespace.
    """
    if not text:
        return ""

    # Lowercase
    text = text.lower()

    # Remove citation brackets (e.g., "[1]" or "[citation needed]")
    text = re.sub(r'\[.*?\]', '', text)

    # Clean up residual wikitext/symbols
    text = re.sub(r'["“”‘’\'\']', '', text)
    text = text.replace('\\', '').replace(':', '').replace('-', ' ')

    # Unicode Character Normalization (e.g., 'ō' to 'o')
    normalized_value = unicodedata.normalize('NFKD', text)
    text = normalized_value.encode('ascii', 'ignore').decode('utf-8')

    # Final Whitespace Normalization (including removing leading/trailing punctuation)
    text = re.sub(r'\s+', ' ', text).strip().strip('.,')

    return text


def parse_slang(raw_text):
    """
    Parses the raw slang text into a dictionary of term: definition pairs,
    handling multiple terms and splitting on the colon separator.
    """
    parsed_slang = {}

    # Split text by line and clean up extra blank lines/headers
    lines = [line.strip() for line in raw_text.split('\n') if line.strip() and ':' in line]

    for line in lines:
        try:
            # Find the first colon to split the term from the definition
            term_part, definition_part = line.split(':', 1)

            # Split terms that use 'or' or '/' (e.g., 'Doughboy or Doughgirl')
            raw_terms = re.split(r'\s*or\s*|\s*/\s*', term_part.strip())

            # Clean the definition once
            cleaned_definition = normalize_text(definition_part.strip())

            if not cleaned_definition:
                continue

            for raw_term in raw_terms:
                if raw_term:
                    cleaned_term = normalize_text(raw_term)

                    if cleaned_term and cleaned_term not in parsed_slang:
                        # Store the cleaned term and its cleaned definition
                        # For embedding, we will use the clean term as the key
                        parsed_slang[cleaned_term] = cleaned_definition

        except ValueError:
            # Skip lines that don't conform to the Term: Definition format
            continue

    return parsed_slang


def create_document_for_embedding(article_data):
    """
    Concatenates all relevant key-value pairs from a single article's dictionary
    into a single string for the 'summary' field, and returns a dict with 'title' and 'summary'.
    """
    doc_parts = []

    # Extract the title separately
    article_title = article_data.get('title', 'Unknown Title')

    # Add the title as the first part of the summary for strong contextual signal
    if article_title:
        doc_parts.append(f"title: {article_title}")

    for key, value in article_data.items():
        # Skip irrelevant keys and the title (already added)
        if key == 'title' or key in KEYS_TO_SKIP:
            continue

        # Ensure value is a string (or convert list of strings to string)
        if isinstance(value, list):
            value = " ".join(value)
        elif not isinstance(value, str):
            value = str(value)

        # Only add if the value is not empty after conversion/joining
        if value.strip():
            # Format: "key: value." (keeping the structure for embedding)
            doc_parts.append(f"{key}: {value}")

    # Join all parts with a period and space for clear sentence separation
    merged_summary = ". ".join(doc_parts) + "."

    # Final document structure with only two keys
    return {
        'title': article_title,
        'summary': merged_summary
    }


def clean_timeline_text(text):
    """Performs cleaning and normalization for timeline entries."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove citation brackets (e.g., "[ 1 ]" or "[citation needed]")
    text = re.sub(r'\[.*?\]', '', text)

    # 3. Remove URLs/Web links (though less likely here, it's good practice)
    text = re.sub(r'https?:\/\/\S+|www\.\S+', '', text, flags=re.IGNORECASE)

    # 4. Final Whitespace Normalization
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# # --- Execution (Timeline) ---
# try:
#     with open(TIMELINE_INPUT_FILE, 'r', encoding='utf-8') as f:
#         timeline_data = json.load(f)
# except FileNotFoundError:
#     raise FileNotFoundError(f"Error: Could not find the required file: {TIMELINE_INPUT_FILE}")
#
# timeline_embedding_documents = {}
# count = 0
#
# for item in timeline_data:
#     year = clean_timeline_text(item.get('year', ''))
#     event = clean_timeline_text(item.get('event', ''))
#
#     if year and event:
#         # **UPDATED:** NEW STRUCTURE to emphasize the event
#         merged_summary = f"Timeline Event: {event}. The year this occurred was: {year}."
#
#         # Use a unique identifier (year + index) as the key/document ID
#         doc_id = f"{year}_{count}"
#
#         # New structure: dictionary with title and summary
#         timeline_embedding_documents[doc_id] = {
#             'title': f'Timeline Event {year}',
#             'summary': merged_summary
#         }
#         count += 1
#
# # Save the final result
# with open(TIMELINE_OUTPUT_FILE, 'w', encoding='utf-8') as f:
#     json.dump(timeline_embedding_documents, f, ensure_ascii=False, indent=4)
#
# # Print a sample
# sample_keys = list(timeline_embedding_documents.keys())[:3]
# sample_output = {key: timeline_embedding_documents[key] for key in sample_keys}
#
# print(f"Total timeline entries prepared for embedding: {len(timeline_embedding_documents)}")
# print(f"Timeline embedding documents saved to: {TIMELINE_OUTPUT_FILE}")
# print("\n--- Sample of Timeline Documents Ready for Embedding (First 3 Entries) ---")
# print(json.dumps(sample_output, indent=4, ensure_ascii=False))
#
# # --- Execution (Main Corpus) ---
# try:
#     with open(INPUT_FILE, 'r', encoding='utf-8') as f:
#         normalized_corpus = json.load(f)
# except FileNotFoundError:
#     raise FileNotFoundError(f"Error: Could not find the required file: {INPUT_FILE}")
#
# embedding_documents = {}
# for title, article_data in normalized_corpus.items():
#     if article_data:
#         # doc_info = {'title': ..., 'summary': ...}
#         doc_info = create_document_for_embedding(article_data)
#
#         # Store as {ID: {'title': ..., 'summary': ...}}
#         if doc_info['summary'].strip() != '.':  # Check if content was actually created
#             embedding_documents[title] = doc_info
#
#         # Save the final result
# with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#     json.dump(embedding_documents, f, ensure_ascii=False, indent=4)
#
# # Print a sample
# sample_keys = list(embedding_documents.keys())[:2]
# sample_output = {key: embedding_documents[key] for key in sample_keys}
#
# print(f"Total documents prepared for embedding: {len(embedding_documents)}")
# print(f"Embedding documents saved to: {OUTPUT_FILE}")
# print("\n--- Sample of Document Text Ready for Embedding (First 2 Articles) ---")
# print(json.dumps(sample_output, indent=4, ensure_ascii=False))
#

# NER
# --- Main Execution ---
ner_corpus = {}
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    total_docs = len(corpus)
    print(f"Starting NER on {total_docs} documents...")

    for i, (title, data) in enumerate(corpus.items()):
        # We need to construct a single string of ALL text from the document for NER
        text_to_process = ""
        for key, value in data.items():
            if key in ['title', 'template_type']:  # Skip structural keys
                continue

            # Ensure value is string and append
            if isinstance(value, list):
                text_to_process += " ".join(value) + " "
            elif isinstance(value, str):
                text_to_process += value + " "

        if not text_to_process.strip():
            continue

        entities = extract_entities(text_to_process)

        # Store the extracted entities under the document title
        # Store the list of entity objects, NOT the dictionary keys
        ner_corpus[title] = entities

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{total_docs} documents...")

    # Save the final NER results
    with open(NER_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(ner_corpus, f, ensure_ascii=False, indent=4)

    print(f"\nNER analysis complete! Extracted entities for {len(ner_corpus)} documents.")
    print(f"Output saved to: {NER_OUTPUT_FILE}")

except FileNotFoundError:
    print(f"Error: Input file {INPUT_FILE} not found. Please ensure it exists in the same directory.")


# --- Execution ---
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        normalized_corpus = json.load(f)
except FileNotFoundError:
    print(f"Error: Input file {INPUT_FILE} not found.")
    exit()

document_type_map = {}
for title, article_data in normalized_corpus.items():
    if article_data:
        doc_type = get_document_type(article_data)
        document_type_map[title] = doc_type

# Save the final result
with open(MAPPING_OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(document_type_map, f, ensure_ascii=False, indent=4)

print(f"Document Type Map created for {len(document_type_map)} documents.")
print(f"File saved to: {MAPPING_OUTPUT_FILE}")
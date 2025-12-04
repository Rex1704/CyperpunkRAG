import json
import re
import unicodedata

RAW_FILE_NAME = "data/cyberpunk_fandom_data_api.json"
OUTPUT_FILE_NAME = "data/cyberpunk_structured_data.json"
KEYS_TO_SKIP = ['seealso', 'references', 'source', 'additional']


def clean_wikilinks(text):
    """ Cleans wikitext links: [[Target|Display Text]] -> Display Text, [[Target]] -> Target """
    if not text: return ""
    # 1. [[Target Page|Display Text]] -> Display Text
    text = re.sub(r'\[\[[^|\]]+\|([^\]]+)\]\]', r'\1', text)
    # 2. [[Target Page]] -> Target Page
    text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)
    # 3. Remove list markers, excessive spacing
    text = text.replace('*', '').strip()
    # Remove HTML tags/comments
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_wikitext_for_structured_data(wikitext):
    """ Combines template and section parsing into a single dictionary output. """
    data = {}

    # --- 1. PRIMARY TEMPLATE EXTRACTION (e.g., {{Shard...}} or {{Infobox...}}) ---
    # Find the first major template and its content, assuming it's near the start
    template_match = re.search(r'\{\{([^\n|]+)\s*[\n](.*?)}}', wikitext, re.DOTALL)

    if template_match:
        template_name = template_match.group(1).strip()
        data['template_type'] = template_name

        # Regex for key-value extraction within the template (multi-line, non-greedy)
        template_content = template_match.group(2)
        kv_pattern = re.compile(r'\|\s*(\w+)\s*=\s*(.*?)(?=\|\s*\w+\s*=|}})', re.DOTALL)

        for match in kv_pattern.finditer(template_content):
            key = match.group(1).strip().lower()
            value = match.group(2).strip()
            data[key] = value

        # Remove the processed template block and surrounding template invocation from the wikitext
        wikitext = wikitext[template_match.end():].strip()

    # --- 2. INTRO/OVERVIEW TEXT EXTRACTION ---
    # Everything before the first == Header == or category/interwiki links is the intro text
    intro_match = re.match(r'(.*?)(?=\n==|\n\[\[Category:|\n\[\[\w{2}:|\Z)', wikitext, re.DOTALL)

    if intro_match:
        intro_text = intro_match.group(1).strip()
        if intro_text:
            # Check if this text is just leftover small templates or boilerplate and filter if so
            if len(intro_text) > 50 and not intro_text.startswith('{'):
                data['overview_description'] = intro_text

        # Remove the intro block from the wikitext
        wikitext = wikitext[intro_match.end():].strip()

    # --- 3. SECTION EXTRACTION (e.g., == Skills ==) ---
    section_pattern = re.compile(r'==\s*(.*?)\s*==\s*(.*?)(?=\n==|\Z)', re.DOTALL)

    for match in section_pattern.finditer(wikitext):
        header = match.group(1).strip().lower().replace(' ', '_')
        content = match.group(2).strip()

        # Only add section if it doesn't conflict with core template data
        if header not in data:
            data[header] = content

    # --- 4. FINAL CLEANUP AND FILTERING ---
    final_data = {}

    for key, value in data.items():
        if key in KEYS_TO_SKIP or not value:
            continue

        # Clean wikitext links and normalize text
        cleaned_value = clean_wikilinks(value)

        if cleaned_value:
            # Standardize key names and filter categories/interwiki
            if key == 'title':
                final_data[key] = cleaned_value
            elif key == 'transcript':
                final_data['transcript_log'] = cleaned_value
            else:
                # Catch remaining categories/interwiki links
                cleaned_value = re.sub(r'\[\[Category:.*?\]\]', '', cleaned_value)
                cleaned_value = re.sub(r'\[\[\w{2}:.*?\]\]', '', cleaned_value)
                final_data[key] = cleaned_value

    return final_data


# --- Execution ---
try:
    with open(RAW_FILE_NAME, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    # This should not happen as the file was uploaded, but defensive programming is good.
    raise FileNotFoundError(f"Error: The file {RAW_FILE_NAME} was not found.")

structured_corpus = {}
for title, wikitext in raw_data.items():
    # 1. Filter out redirects
    if wikitext and isinstance(wikitext, str) and wikitext.strip().upper().startswith('#REDIRECT'):
        continue

    # 2. Process article
    clean_title = title.replace('_', ' ').strip()

    if isinstance(wikitext, str):
        processed_data = parse_wikitext_for_structured_data(wikitext)

        # Ensure title is present, using filename title if template title is missing
        if 'title' not in processed_data or not processed_data['title']:
            processed_data['title'] = clean_title

        if processed_data.get('title'):
            structured_corpus[processed_data['title']] = processed_data

OUTPUT_NORMALIZED_FILE = "data/cyberpunk_fully_normalized_corpus.json"


# --- 1. Define Enhanced Cleaning Function ---

def final_cleaning_and_normalization(value):
    """
    Performs URL removal, lowercasing, quote cleanup, and special character
    (Unicode/escape sequence) normalization.
    """
    if not isinstance(value, str):
        return value

    # 1. Lowercase (Done first to simplify subsequent regex)
    value = value.lower()

    # 2. Quote and Escape Sequence Cleanup:
    # Remove backslashes used for JSON escaping (if any slipped through)
    value = value.replace('\\', '')

    # Remove escaped/special quotes ("...", '...', and unicode fancy quotes)
    # The \" in the title examples is handled by the initial backslash removal.
    # The list covers common straight/curly quotes and apostrophes.
    value = re.sub(r'["“”‘’\'\']', '', value)

    # 3. URL Removal
    value = re.sub(r'https?:\/\/\S+|www\.\S+|\[\s*http[^\s]*\s*([^\]]*)\]', r'\1', value, flags=re.IGNORECASE)

    # 4. Unicode Character Normalization (e.g., 'ō' to 'o')
    # a) Decompose characters (e.g., 'o' + macron combiner)
    normalized_value = unicodedata.normalize('NFKD', value)
    # b) Encode to ASCII, ignoring errors (discarding characters that have no close ASCII equivalent)
    # This addresses the 'ō' to 'o' requirement and handles most other non-english characters
    value = normalized_value.encode('ascii', 'ignore').decode('utf-8')

    # 5. Final Whitespace Normalization
    value = re.sub(r'\s+', ' ', value).strip()

    return value


def deep_clean_and_normalize(data):
    """Recursively applies final_cleaning_and_normalization to all string values."""
    if isinstance(data, dict):
        # Recursively apply to dictionary values
        return {k: deep_clean_and_normalize(v) for k, v in data.items()}
    elif isinstance(data, list):
        # Recursively apply to list elements
        return [deep_clean_and_normalize(i) for i in data]
    elif isinstance(data, str):
        # Apply cleaning to string values
        return final_cleaning_and_normalization(data)
    else:
        # Return non-string/non-container types unchanged
        return data


# --- 2. Load and Process Data ---
try:
    with open(OUTPUT_FILE_NAME, 'r', encoding='utf-8') as f:
        structured_data = json.load(f)
except FileNotFoundError:
    # Fallback to the user's initial file if the structured one is missing (should not happen here)
    raise FileNotFoundError(f"Error: Could not find the required structured file: {OUTPUT_FILE_NAME}")

# Apply deep cleaning to the entire corpus
final_normalized_corpus = deep_clean_and_normalize(structured_data)

# --- 3. Save Output and Provide Sample ---
with open(OUTPUT_NORMALIZED_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_normalized_corpus, f, ensure_ascii=False, indent=4)

# Print a sample
# The title should now be 'bushido and neopostmodernism'
target_title_key = "bushido and neopostmodernism"
sample_output = {}

# Iterate to find the normalized title key
for original_key, data in final_normalized_corpus.items():
    if target_title_key in original_key:
        sample_output[original_key] = data
        break

print(f"Total articles cleaned and saved: {len(final_normalized_corpus)}")
print(f"Final normalized corpus saved to: {OUTPUT_NORMALIZED_FILE}")
print("\n--- Sample of Final Cleaned and Normalized Corpus (Target Article) ---")
print(json.dumps(sample_output, indent=4, ensure_ascii=False))

# # Save the final result
# with open(OUTPUT_FILE_NAME, 'w', encoding='utf-8') as f:
#     json.dump(structured_corpus, f, ensure_ascii=False, indent=4)

# # Print a sample
# sample_keys = list(structured_corpus.keys())[:2]
# sample_output = {key: structured_corpus[key] for key in sample_keys}
#
# print(f"Total articles structured and saved: {len(structured_corpus)}")
# print(f"Structured data saved to: {OUTPUT_FILE_NAME}")
# print("\n--- Sample of Combined Structured Data (First 2 Articles) ---")
# print(json.dumps(sample_output, indent=4, ensure_ascii=False))


import requests
import time
import json
import re
import mwparserfromhell as mwp

# --- CONFIGURATION ---
BASE_API_URL = "https://cyberpunk.fandom.com/api.php"
OUTPUT_FILENAME = "data/cyberpunk_fandom_data_api.json"
# Add a delay to be a polite scraper and avoid potential IP blocks
REQUEST_DELAY_SECONDS = 0.5


# --- 1. FUNCTION TO GET ALL ARTICLE TITLES ---
def get_all_article_titles():
    """
    Uses the 'allpages' list module to retrieve all page titles from the wiki,
    handling pagination automatically.
    """
    print("🚀 Starting API query to collect ALL article titles...")

    titles = []
    apcontinue = None  # Used for pagination

    while True:
        params = {
            'action': 'query',
            'list': 'allpages',
            'aplimit': 'max',  # Request the maximum number of titles per page (500)
            'format': 'json',
            'apnamespace': 0,  # Filter for main articles only (Namespace 0)
        }

        # Add the continuation parameter for the next page if it exists
        if apcontinue:
            params['apcontinue'] = apcontinue

        try:
            response = requests.get(BASE_API_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            current_pages = data.get('query', {}).get('allpages', [])

            # Extract and store titles
            for page in current_pages:
                titles.append(page['title'])

            print(f"   -> Collected {len(titles)} titles so far...")

            # Check for the continuation marker for the next page
            apcontinue = data.get('continue', {}).get('apcontinue')

            if not apcontinue:
                break  # Exit the loop when there are no more pages

            time.sleep(REQUEST_DELAY_SECONDS)  # Wait before fetching the next page of titles

        except requests.exceptions.RequestException as e:
            print(f"🛑 Error during title collection: {e}")
            break

    print(f"✅ Finished collecting all titles. Found a total of {len(titles)} articles.")
    return titles


# --- 2. FUNCTION TO FETCH CONTENT FOR A BATCH OF TITLES ---
def get_content_for_titles(titles_list):
    """
    Fetches the raw wikitext content for a list of titles.
    MediaWiki API can handle up to 50 titles per request.
    """
    # Join titles with the pipe character '|' for batch fetching
    title_string = '|'.join(titles_list)

    params = {
        'action': 'query',
        'titles': title_string,
        'prop': 'revisions',
        'rvprop': 'content',  # Request the raw wikitext content
        'format': 'json',
        'apnamespace': 0  # Ensure we're only getting main articles
    }

    try:
        response = requests.get(BASE_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        # The result is nested and keys are page IDs
        pages = data.get('query', {}).get('pages', {})

        results = {}
        for page_id, page_data in pages.items():
            # Skip invalid/missing pages
            if page_id == '-1':
                continue

            title = page_data.get('title')

            # Extract the raw wikitext content
            wikitext = page_data.get('revisions', [{}])[0].get('*')

            if wikitext:
                results[title] = wikitext

        return results

    except requests.exceptions.RequestException as e:
        print(f"🛑 Error fetching content for batch: {e}")
        return {}


# --- 3. MAIN EXECUTION ---
def main_api_crawler():
    # Step 1: Get ALL the page titles
    all_titles = get_all_article_titles()

    # Define the batch size (API limit is 50, but 40-50 is safe)
    BATCH_SIZE = 50
    final_data = {}

    print("\n📦 Starting content fetching in batches...")

    # Step 2: Iterate through all titles in batches
    for i in range(0, len(all_titles), BATCH_SIZE):
        batch = all_titles[i:i + BATCH_SIZE]

        # Fetch content for the current batch
        batch_results = get_content_for_titles(batch)
        final_data.update(batch_results)

        print(f"   -> Progress: {i + len(batch)} / {len(all_titles)} articles fetched.")

        # Be mindful of the server load
        time.sleep(REQUEST_DELAY_SECONDS)

    # Step 3: Save the final data
    print(f"\n✅ All content collected. Total articles: {len(final_data)}")

    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        # Save only the first 100 articles for a sample, or remove  for the full set
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"💾 Data saved to {OUTPUT_FILENAME}")

# Load the JSON data
with open(OUTPUT_FILENAME, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)


# --- Wikitext Cleaning Function ---
def clean_wikitext(text):
    """
    Cleans raw MediaWiki wikitext by removing templates, categories,
    references, tables, and formatting tags.
    """

    if text is None:
        return ""
    parser = mwp.parser.Parser()

    wikicode = parser.parse(text)

    return wikicode.strip()

    # 1. Handle Redirects: If the whole content is a redirect, just return the redirect target.
    redirect_match = re.match(r'#REDIRECT\s*\[\[(.*?)\]\]', text.strip(), re.IGNORECASE)
    if redirect_match:
        # Return a concise text indicating the redirect target
        return f"REDIRECT_TO: {redirect_match.group(1)}"

    # 2. Remove Categories (e.g., [[Category:Cyberpunk V3.0 Characters]])
    text = re.sub(r'\[\[Category:.*?\]\]', '', text)

    # 3. Remove all Templates (e.g., {{Infobox character|...}}, {{GameIcon|2077}})
    # Non-greedy matching for the first template pair, looping to handle non-nested multiple templates
    while '{{' in text and '}}' in text:
        text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)

    # Remove HTML comments
    text = re.sub(r'', '', text, flags=re.DOTALL)

    # 4. Remove Tables (e.g., {| ... |} )
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)

    # 5. Simplify Internal/External Links: keep only the display text
    # [[Target Page|Display Text]] -> Display Text
    text = re.sub(r'\[\[[^|\]]+\|([^\]]+)\]\]', r'\1', text)
    # [[Target Page]] -> Target Page
    text = re.sub(r'\[\[(.*?)\]\]', r'\1', text)
    # External Links [http://... Display Text] -> Display Text
    text = re.sub(r'\[https?://[^ ]+\s*([^\]]*)\]', r'\1', text)
    # External Links [http://...] -> remove completely
    text = re.sub(r'\[https?://[^ ]+\s*\]', r'', text)

    # 6. Remove remaining Wikitext formatting (bold/italics)
    text = re.sub(r"'''|''", '', text)

    # 7. Remove remaining HTML tags (e.g., <br />, <div>)
    text = re.sub(r'<[^>]+>', '', text)

    # 8. Clean up section headers (e.g., == Skills ==)
    text = re.sub(r'={2,}\s*(.*?)\s*={2,}', r'\n\n-- \1 --\n', text)

    # 9. Clean up multiple newlines and leading/trailing whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()

    return text


def clean_wikitext_mwp(wikitext):
    """
    Robustly clean wikitext to plain text, ignore malformed templates.
    """
    # Check for redirect
    redirect_match = re.match(r"#REDIRECT\s*\[\[(.*?)\]\]", wikitext, re.IGNORECASE)
    if redirect_match:
        return {
            "plain_text": "",
            "sections": [],
            "is_redirect": True,
            "redirect_target": redirect_match.group(1)
        }

    try:
        wikicode = mwp.parse(wikitext)
    except Exception as e:
        # Fallback: just strip known template patterns using regex
        wikicode = wikitext
        wikicode = re.sub(r"\{\{.*?\}\}", "", wikicode, flags=re.DOTALL)

    # Remove templates safely
    if isinstance(wikicode, mwp.wikicode.Wikicode):
        try:
            for template in wikicode.filter_templates(recursive=True):
                try:
                    wikicode.remove(template)
                except:
                    continue
        except:
            pass

        # Remove comments
        for comment in wikicode.filter_comments():
            try:
                wikicode.remove(comment)
            except:
                continue

        # Remove tags (like <ref>...</ref>)
        for tag in wikicode.filter_tags():
            try:
                wikicode.remove(tag)
            except:
                continue

        # Convert to plain text
        plain_text = wikicode.strip_code().strip()
    else:
        # fallback string
        plain_text = str(wikicode)
        plain_text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", plain_text)
        plain_text = re.sub(r"''+", "", plain_text)

    # Normalize whitespace
    plain_text = " ".join(plain_text.split())

    # Sections extraction
    sections = []
    if isinstance(wikicode, mwp.wikicode.Wikicode):
        for section in wikicode.get_sections(include_lead=True, levels=[2,3]):
            headings = section.filter_headings()
            heading = headings[0].title.strip() if headings else "Intro"
            text = section.strip_code().strip()
            text = " ".join(text.split())
            sections.append({"heading": heading, "text": text})

    return {
        "plain_text": plain_text,
        "sections": sections,
        "is_redirect": False,
        "redirect_target": None
    }

# --- Save and Preview ---


if __name__ == "__main__":
    # main_api_crawler()
    # --- Process the Data ---




    cleaned_data = {}
    for title, wikitext in raw_data.items():
        cleaned_text = clean_wikitext_mwp(wikitext)

        # Store clean text only if it's not empty after cleaning
        if cleaned_text:
            # Titles are cleaned up for embedding if they contain encoded characters
            clean_title = title.replace('_', ' ').replace('%22', '"').replace('%C5%8D', 'ō')

            cleaned_data[clean_title] = {
                "title": clean_title,
                "wikitext_content": cleaned_text
            }

    output_file_name = "data/cyberpunk_mwp_cleaned_text.json"
    with open(output_file_name, 'w', encoding='utf-8') as f:
        # Save the full set of cleaned data
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    # Print a sample of the cleaned data
    sample_keys = list(cleaned_data.keys())[:5]
    sample_output = {key: cleaned_data[key] for key in sample_keys}

    print(f"Total articles cleaned: {len(cleaned_data)}")
    print(f"Cleaned data saved to: {output_file_name}")
    print("\n--- Sample of Cleaned Data (First 5 Articles) ---")
    print(json.dumps(sample_output, indent=4, ensure_ascii=False))

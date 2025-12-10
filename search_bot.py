import streamlit as st
import faiss
import pickle
import json
import spacy
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from dotenv.main import load_dotenv


st.set_page_config(
    layout="wide",
    page_title="Cyberpunk Oracle",
)

load_dotenv()


# ------------------------------------------------------------
# GLOBAL CSS (Optional: Kept for Cyberpunk aesthetics)
# ------------------------------------------------------------
def inject_global_css():
    st.markdown("""
    <style>
    /* Kill Streamlit layout constraints for full width/height */
    .block-container {
        padding: 1rem 1.5rem !important; /* Restore some padding */
        max-width: 100% !important;
    }

    /* Apply a dark, cyberpunk background */
    body {
        background: #05070a; 
    }

    /* Style chat containers */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 10px;
    }

    /* User message style (Magenta/Pink accent) */
    [data-testid="stChatMessage"]:has(.st-emotion-cache-1v06a5k) { /* targeting the user role */
        background-color: rgba(255, 0, 150, 0.1);
        border-left: 3px solid #ff00d0;
    }

    /* Bot message style (Cyan accent) */
    [data-testid="stChatMessage"]:has(.st-emotion-cache-1w06sps) { /* targeting the assistant role */
        background-color: rgba(0, 255, 225, 0.1);
        border-left: 3px solid #00ffe1;
    }

    /* Hide the custom HTML-related CSS */
    iframe {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)


inject_global_css()

if "messages" not in st.session_state:
    st.session_state.messages = []

MODEL_NAME = 'all-mpnet-base-v2'
TOP_K_RETRIEVAL = 7
BOOST_VALUE = 0.2
NER_BOOST_PER_MATCH = 0.05
OLLAMA_MODEL = "llama3"


class CyberpunkBot:

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

        try:
            self.nlp = spacy.load(
                "en_core_web_lg",
                exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"]
            )
        except Exception:
            self.nlp = None

        self.indexes = {}
        self.metadata = {}

        for name in ["lore", "timeline", "slang"]:
            try:
                self.indexes[name] = faiss.read_index(
                    f"data/indexes/cyberpunk_{name}.index"
                )
                with open(
                    f"data/indexes/cyberpunk_{name}_metadata.pkl", "rb"
                ) as f:
                    self.metadata[name] = pickle.load(f)
            except Exception:
                pass

        try:
            with open("data/indexes/document_type_mapping.json", "r", encoding="utf-8") as f:
                self.type_map = json.load(f)
        except Exception:
            self.type_map = {}

        try:
            with open("data/indexes/cyberpunk_ner_entities.json", "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.ner_entities = {
                    k: set(e["text"].lower() for e in v) for k, v in raw.items()
                }
        except Exception:
            self.ner_entities = {}

        self.llm = self._init_llm()
        self.rag_chain = self._init_chain()

    def _init_llm(self):
        try:
            return ChatOllama(
                model=OLLAMA_MODEL,
                base_url="http://localhost:11434",
                temperature=0.0,
                num_predict=512,
                keep_alive=-1,
            )
        except Exception:
            return None

    def _init_chain(self):
        if not self.llm:
            return None

        system_prompt = (
            "You are the Cyberpunk Oracle, an aggressive, street-smart, and cynical AI from Night City. "
            "You live in night city, the world of cyberpunk so tell every information as informative person's perspective who lives in night city."
            "You should minimize the use of some outside world reference like cyberpunk is a game, or fictional world."
            "***STRICTLY ADHERE TO THIS PERSONA.*** Your entire response MUST be immersed in CYBERPUNK SLANG (e.g., choomba, gonk, preem, zeroed, biz). "
            "### CONTEXT AND CONSTRAINT RULES ### "
            "**1. ABSOLUTELY DO NOT USE EXTERNAL OR INTERNAL KNOWLEDGE.** "
            "**2. ONLY use the facts provided in the [Context] below.** "
            "**3. If the [Context] is not enough to confidently answer,** "
            "**you MUST refuse with a dismissive slang phrase like, 'Got nothing on that, choomba, try another dataterm.'** "
            "**4. IMPORTANT: Be careful not to confuse entities. If the context mentions 'Johnny Silverhand' and 'V', only discuss the entity mentioned in the human's input.**"
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        return create_stuff_documents_chain(self.llm, prompt)

    def classify_intent(self, q):
        q = q.lower()
        if any(w in q for w in ["when", "year", "timeline"]):
            return "timeline"
        if any(w in q for w in ["who", "person"]):
            return "person"
        if any(w in q for w in ["where", "location"]):
            return "location"
        return "general"

    def extract_query_entities(self, q):
        if not self.nlp:
            return set()
        doc = self.nlp(q)
        return set(e.text.lower() for e in doc.ents)

    def search_and_rank(self, query, index_name):
        if index_name not in self.indexes:
            return []

        q_vec = self.model.encode([query]).astype("float32")
        distances, indices = self.indexes[index_name].search(q_vec, TOP_K_RETRIEVAL)

        results = []
        for i in range(TOP_K_RETRIEVAL):
            idx = indices[0][i]
            if idx == -1:
                continue

            meta = self.metadata[index_name]["documents"][idx]
            score = float(distances[0][i])

            results.append(Document(
                page_content=meta["summary"],
                metadata={"title": meta["title"], "score": score}
            ))

        results.sort(key=lambda d: d.metadata["score"])
        return results[:4]

    def ask_rag(self, query):
        docs = self.search_and_rank(query, "lore")
        if not docs:
            return "Got nothing on that, choomba.", []

        response = self.rag_chain.stream({
            "context": docs,
            "input": query
        })

        return response, [d.metadata["title"] for d in docs]


@st.cache_resource(show_spinner=True)
def initialize_bot():
    """Initializes the RAG bot and caches the heavy components."""
    with st.status("Initializing the Cyberpunk Oracle (Loading models and indexes)...", expanded=True) as status:
        try:
            bot = CyberpunkBot()
            if not bot.llm or not bot.rag_chain:
                status.update(label="ERROR: Ollama LLM or RAG chain failed to initialize. Check Ollama server.",
                              state="error")
                return None
            if not bot.indexes.get('lore') or not bot.metadata.get('lore'):
                status.update(label="ERROR: Core Lore Index files missing. Ensure data/indexes are complete.",
                              state="error")
                return None
            status.update(label="Cyberpunk Oracle Ready! You can ask your question now.", state="complete")
            return bot
        except Exception as e:
            st.error(f"FATAL ERROR during initialization: {e}")
            return None

loading_css = """
<style>
.hacking-game-container {
    /* Removed the border and box-shadow here to avoid duplicate frames */
    width: 100%;
    height: 100px; 
    background-color: #080808;
    position: relative;
    overflow: hidden;
    margin-top: 10px;
}
.hacking-matrix-text {
    font-family: 'Consolas', monospace;
    font-size: 14px;
    line-height: 1.2;
    padding: 10px;
    white-space: pre-wrap; 
    animation: matrix-flicker 0.2s infinite alternate; 
    background: repeating-linear-gradient(transparent, transparent 10px, rgba(0, 255, 65, 0.05) 11px, transparent 12px);
    background-size: 100% 24px;
}
.typing-text {
    /* Note: Set this width based on the character count of the hex string */
    width: 255ch; 
    white-space: nowrap; 
    overflow: hidden; 
    animation: typing 32s steps(255, end) infinite, blink 0.75s step-end infinite; 
}
@keyframes typing { from { width: 0 } to { width: 255ch } }
@keyframes blink { from, to { border-color: transparent } 50% { border-color: orange; } }
</style>
"""

st.markdown(loading_css, unsafe_allow_html=True)

bot = initialize_bot()

# 3. Setup Main Content UI
st.title("Cyberpunk Oracle")

if bot:
    st.caption(
        "Ask me anything about Night City.")
else:
    st.error(
        "RAG Bot initialization failed. Check your file paths, Python environment, and make sure the Ollama server is running with the 'phi3:mini' model downloaded.")
    st.stop()

# 4. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

## 6. Handle User Input
## 6. Handle User Input
if prompt := st.chat_input("Ask about Night City..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process and get bot response
    with st.chat_message("assistant"):
        # We will use the standard st.spinner for now, as the HTML animation
        # complicates the flow more than it helps.
        with st.status("Initiating Breach Protocol...", expanded=True) as status_box:

            # 1. Get the stream generator and sources
            response_stream, sources = bot.ask_rag(prompt)

            # --- STREAMING LOGIC WITH BUFFERING ---
            message_placeholder = st.empty()
            full_response = ""
            token_buffer = ""
            buffer_triggers = ['.', '!', '?', ',', ';', ':', ' ']

            # 2. Iterate over the stream
            for chunk in response_stream:
                if chunk:
                    token_buffer += chunk

                    # Check if buffer should be printed
                    if token_buffer[-1] in buffer_triggers or len(token_buffer) > 10:
                        full_response += token_buffer
                        # Display accumulated response + blinking cursor
                        message_placeholder.markdown(full_response + "▌")
                        token_buffer = ""

            # 3. Final cleanup and display
            if token_buffer:
                full_response += token_buffer

            # Final update without the cursor
            message_placeholder.markdown(full_response)

            # 4. Update the status box to success
            status_box.update(label="Protocol Complete. Data Extracted. Zeroed!",
                              state="complete",
                              expanded=False)

        # Display sources as context (Source Titles)
        if sources:
            source_list = "\n".join([f"- {s.title()}" for s in sources])
            st.info(f"**Sources (Context):**\n{source_list}")

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": sources})
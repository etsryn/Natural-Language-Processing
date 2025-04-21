import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import streamlit as st
# Set page config as the first command (only once)
st.set_page_config(page_title="🧠 NLP Playground", layout="wide")
# ====== Streamlit UI ======
st.title("🧠 NLP Playground")

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, regexp_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
import spacy
from spacy import displacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease

from transformers import pipeline

# ====== Setup & Caching ======
@st.cache_resource
def load_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

from transformers import pipeline

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")


# Initialize resources
load_nltk_data()
nlp = load_spacy_model()
summarizer = load_summarizer()
analyzer = SentimentIntensityAnalyzer()
stops = set(stopwords.words('english'))
porter = PorterStemmer()
snowball = SnowballStemmer('english')
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

# ====== Helper Functions ======

def get_wordnet_pos(tag):
    tag_initial = tag[0].upper()
    return {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }.get(tag_initial, wordnet.NOUN)

def tokenize(text, method='nltk'):
    if method == 'nltk':
        return word_tokenize(text)
    if method == 'regex':
        return regexp_tokenize(text, pattern="\\w+")
    # spaCy
    return [token.text for token in nlp(text)]


# Sidebar inputs
text = st.sidebar.text_area("Enter text:", height=100)
if not text.strip():
    st.sidebar.button("Analyse")
    st.stop()

# Tabs for phases
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Lexical & Morphological",
    "Syntactic",
    "Semantic",
    "Discourse",
    "Pragmatic"
])

# ====== Tab 1: Lexical & Morphological ======
with tab1:
    st.header("🔤 Lexical & Morphological Analysis")
    col1, col2 = st.columns(2)
    # Tokenization
    with col1:
        st.subheader("1. Tokenization")
        method = st.selectbox("Method:", ["nltk", "regex", "spacy"])
        if st.button("Run Tokenization", key="tok"):
            tokens = tokenize(text, method)
            st.write(tokens)

    # Stopwords & Stemming / Lemmatization
    with col2:
        st.subheader("2. Stopwords Removal")
        if st.button("Remove Stopwords", key="stop"):
            filtered = [w for w in tokenize(text) if w.lower() not in stops]
            st.write(filtered)

        st.subheader("3. Stemming")
        stemmer_choice = st.radio("Stemmer:", ["Porter", "Snowball", "Lancaster"])
        if st.button("Stem", key="stem"):
            stem_map = {"Porter": porter, "Snowball": snowball, "Lancaster": lancaster}
            stems = [stem_map[stemmer_choice].stem(w) for w in tokenize(text)]
            st.write(stems)

        st.subheader("4. Lemmatization")
        if st.button("Lemmatize", key="lem"):
            lemmas = []
            for w, tag in nltk.pos_tag(tokenize(text)):
                lemmas.append(lemmatizer.lemmatize(w, get_wordnet_pos(tag)))
            st.write(lemmas)

# ====== Tab 2: Syntactic ======
with tab2:
    st.header("🔗 Syntactic Analysis")
    # POS & Dependency
    pos_col, dep_col = st.columns(2)
    with pos_col:
        st.subheader("POS Tagging")
        lib = st.selectbox("Library:", ["nltk", "spaCy"], key="pos_lib")
        if st.button("Tag POS", key="pos"):
            if lib == 'nltk':
                st.write(nltk.pos_tag(tokenize(text)))
            else:
                st.write([(t.text, t.pos_) for t in nlp(text)])
    with dep_col:
        st.subheader("Dependency Plot")
        if st.button("Show Parse", key="dep"):
            doc = nlp(text)
            html = displacy.render(doc, style='dep', jupyter=False, options={'compact': True})
            st.write(html, unsafe_allow_html=True)

# ====== Tab 3: Semantic ======
with tab3:
    st.header("🧠 Semantic Analysis")
    # Bag-of-Words & TF-IDF
    vec_choice = st.selectbox("Vectorizer:", ["CountVectorizer", "TfidfVectorizer"], key="vec")
    if st.button("Vectorize", key="vec_btn"):
        vec = CountVectorizer() if vec_choice=='CountVectorizer' else TfidfVectorizer()
        X = vec.fit_transform([text])
        st.dataframe({"Feature": vec.get_feature_names_out(), "Score": X.toarray()[0]})

    # Embeddings (spaCy)
    if st.button("Show Word Embeddings", key="embed"):
        doc = nlp(text)
        embed_list = [{token.text: token.vector.tolist()} for token in doc if token.has_vector]
        st.write(embed_list)

# ====== Tab 4: Discourse ======
with tab4:
    st.header("📰 Discourse Integration")
    if st.button("Segment Sentences", key="disc_sent"):
        st.write(sent_tokenize(text))

    if st.button("Named Entities", key="disc_ner"):
        ents = [(e.text, e.label_) for e in nlp(text).ents]
        st.write(ents)

    num_sents = st.slider("Summary sentences:", 1, 5, 3, key="sum_slider")
    if st.button("Summarize", key="disc_sum"):
        summary = summarizer(text, max_length=num_sents*30, min_length=num_sents*10)
        st.write(summary[0]['summary_text'])

# ====== Tab 5: Pragmatic ======
with tab5:
    st.header("🎯 Pragmatic Analysis")
    if st.button("TextBlob Sentiment", key="prag_tb"):
        from textblob import TextBlob
        tb = TextBlob(text)
        st.write(tb.sentiment)
    if st.button("VADER Sentiment", key="prag_vader"):
        st.write(analyzer.polarity_scores(text))
    if st.button("Readability (Flesch)", key="prag_read"):
        try:
            score = flesch_reading_ease(text)
            st.write(f"Flesch Reading Ease: {score:.2f}")
        except ImportError:
            st.error("Install 'textstat' for readability metrics.")

# ====== Footer ======
st.sidebar.button("Analyse")
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ by Rayyan Ashraf")

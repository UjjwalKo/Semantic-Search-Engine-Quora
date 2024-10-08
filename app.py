import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer, util
import time

st.set_page_config(page_title="Quora Question Search", page_icon="❓", layout="centered")
data = joblib.load('embeddings_model.joblib')
embeddings, titles = data
model = SentenceTransformer('LaBSE')

def search_question(query, threshold=0.3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings)[0]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    top_hits = [hit for hit in hits if hit['score'] >= threshold]
    
    return top_hits

st.title("❓ Semantic Search of Quora Questions")
st.markdown("<h3 style='color: #4CAF50;'>Search for the best matching questions with AI-powered search!</h3>", unsafe_allow_html=True)


with st.spinner('Loading AI-powered search engine...'):
    time.sleep(1)

col1, col2, col3 = st.columns([1, 2, 1])  

with col2:
    query = st.text_input("Enter a question to search", "")
    search_button = st.button("Search", use_container_width=True)

if search_button:
    start_time = time.time()  
    progress_bar = st.progress(0)
    
    for percent_complete in range(1, 101, 20):
        time.sleep(0.1)
        progress_bar.progress(percent_complete)
    
    results = search_question(query)
    end_time = time.time() 
    elapsed_time = end_time - start_time
    st.success(f"Search completed in {elapsed_time:.3f} seconds!")

    if results:
        st.markdown(f"<h4 style='color: #2196F3;'>Top matches for: '{query}'</h4>", unsafe_allow_html=True)
        
        for hit in results[:2]:  
            index = hit['corpus_id']
            st.markdown(f"### **🔹 Question**: {titles[index]}")
            st.markdown(f"<i>Relevance Score</i>: {hit['score']:.3f}", unsafe_allow_html=True)
            st.markdown("---")  
    else:
        st.error("No matching question found.")
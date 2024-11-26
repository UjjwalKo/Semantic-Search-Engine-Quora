import joblib
import time
from django.shortcuts import render
from sentence_transformers import SentenceTransformer, util

data = joblib.load('embeddings/embeddings_model.joblib') 
embeddings, titles = data
model = SentenceTransformer('LaBSE')

def search_question(query, threshold=0.3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings)[0]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    top_hits = [hit for hit in hits if hit['score'] >= threshold]
    return top_hits

def search_view(request):
    results = []
    query = ''
    elapsed_time = 0

    if request.method == "POST":
        query = request.POST.get("query")
        start_time = time.time()
        hits = search_question(query)
        elapsed_time = time.time() - start_time

        results = [
            {
                'question': titles[hit['corpus_id']],  
                'score': hit['score']
            }
            for hit in hits[:2] 
        ]

    context = {
        'query': query,
        'results': results,
        'elapsed_time': elapsed_time,
    }
    return render(request, 'search/search.html', context)
import os
import pypdf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

# Path to PDF
PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Construction Intelligence Platform_ML Predictions.pdf")

@st.cache_resource
def load_rag_documents():
    """
    Load PDF pages and build TF-IDF vectorizer for document retrieval.
    """
    if not os.path.exists(PDF_PATH):
        return [], None, None
        
    reader = pypdf.PdfReader(PDF_PATH)
    pages_text = []
    
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages_text.append({
                'page_num': idx + 1,
                'content': text
            })
            
    # Build TF-IDF index
    corpus = [doc['content'] for doc in pages_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    return pages_text, vectorizer, tfidf_matrix

def retrieve_best_context(query, pages_text, vectorizer, tfidf_matrix):
    """
    Retrieve the most relevant PDF page context for a given query.
    """
    if not pages_text or not vectorizer:
        return "No documentation context available."
        
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = similarities.argsort()[-1]
    
    # Return the content of the best matching page
    best_doc = pages_text[best_idx]
    return f"Context from Page {best_doc['page_num']} of Construction Intelligence Platform:\n{best_doc['content']}"

@st.cache_resource
def load_qwen_model():
    """
    Load Qwen tokenizer and model, caching them so they only load once.
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32
    )
    return tokenizer, model

def generate_qwen_answer(query, context, tokenizer, model):
    """
    Use Qwen to generate an answer based on the query and RAG context.
    """
    system_prompt = (
        "You are an expert Construction Intelligence AI Assistant. "
        "Answer the user's question concisely and professionally based strictly on the provided context. "
        "Keep the tone professional and informative. Do not use any emojis. "
        "If the answer cannot be found in the context, use your built-in knowledge to provide a helpful, factual answer."
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=150,
            temperature=0.2, # low temperature for factual answers
            do_sample=False
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

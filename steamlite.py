import streamlit as st
from haystack_pipeline.pipeline import build_pipeline
from pathlib import Path
from threading import Thread
import queue

# Cache the pipeline to avoid reloading on every interaction
@st.cache_resource
def load_pipeline():
    return build_pipeline(Path(__file__).parent)

# Initialize pipeline and result queue
pipe = load_pipeline()
result_queue = queue.Queue()

def run_pipeline_async(question, result_queue):
    """Thread target function to run pipeline without blocking"""
    try:
        result = pipe.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question}
        })
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)

st.title("Farmbotika Agronomy Chatbot")
question = st.text_input("Ask a farming question:")

if st.button("Submit") and question:
    # Clear previous results
    if 'result' in st.session_state:
        del st.session_state['result']
    
    # Start pipeline in background thread
    Thread(target=run_pipeline_async, args=(question, result_queue)).start()
    
    # Display spinner while processing
    with st.spinner("Analyzing your question..."):
        while not result_queue.qsize():
            st.empty()  # Keep spinner active
            
        # Get result from queue
        result = result_queue.get()
        
        if isinstance(result, Exception):
            st.error(f"Error: {str(result)}")
        else:
            st.session_state.result = result["generator"]["replies"][0].text

# Display final result if available
if 'result' in st.session_state:
    st.success(st.session_state.result)
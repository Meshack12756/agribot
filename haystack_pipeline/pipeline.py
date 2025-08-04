# haystack_pipeline/pipeline.py
import time
from pathlib import Path
import logging
from haystack.components.retrievers import InMemoryEmbeddingRetriever
import pandas as pd
from haystack import Document, Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder

def build_pipeline(project_root: Path):
    #Time it takes to build the pipeline
    start_time = time.time()
    
    """Builds and returns a robust RAG pipeline for FarmBotika Agronomy Q&A."""
    # ----------------------------------------
    # 1. Load CSV and create Documents
    # ----------------------------------------
    csv_path = project_root / "data" / "AgroQA_Dataset.csv"
    df = pd.read_csv(csv_path)

    logging.debug(f"CSV columns: {df.columns.tolist()}")

    documents = []
    for _, row in df.iterrows():
        answer_text = str(row.get("answer", "")).strip()
        question_text = str(row.get("question", "")).strip()
        crop_name = str(row.get("crop", "")).strip()
        if not answer_text:
            continue

        meta = {
            "crop": crop_name,
            "question": question_text,
            "source": "FarmBotika_QA"
        }
        documents.append(Document(content=answer_text, meta=meta))

    # ----------------------------------------
    # 2. Fetch and convert web pages
    # ----------------------------------------
    fetcher = LinkContentFetcher()
    converter = HTMLToDocument()

    urls = [
        "https://www.theorganicfarmer.org",
        "https://www.kephis.org",
    ]

    successful_streams = []
    for url in urls:
        try:
            result = fetcher.run(urls=[url])
            successful_streams.extend(result.get("streams", []))
        except Exception as e:
            logging.warning(f"Skipping URL fetch {url}: {e}")

    if successful_streams:
        try:
            html_docs = converter.run(sources=successful_streams).get("documents", [])
            documents.extend(html_docs)
        except Exception as e:
            logging.warning(f"HTML conversion error: {e}")

    # ----------------------------------------
    # 3. Initialize Document Store & index
    # ----------------------------------------
    document_store = InMemoryDocumentStore()

    seen_ids = set()
    unique_documents = []
    for doc in documents:
        if doc.id not in seen_ids:
            unique_documents.append(doc)
            seen_ids.add(doc.id)

    # ----------------------------------------
    # 4. Build Retriever & Prompt (REMOVED REDUNDANT EMBEDDING)
    # ----------------------------------------
    text_embedder = SentenceTransformersTextEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    doc_embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    doc_embedder.warm_up()

    embedded_docs = doc_embedder.run(documents=unique_documents)["documents"]
    document_store.write_documents(embedded_docs)

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        scale_score=True,
        top_k=5
    )

    template = [
        ChatMessage.from_user(
            """
            You are an agronomy assistant. Use the provided context to answer the question.

            Context:
            {% for document in documents %}
            {{ document.content }}
            {% endfor %}
            Question: {{question}}
            Answer:
            """
        )
    ]
    prompt_builder = ChatPromptBuilder(template=template, required_variables=["documents", "question"])

    # ----------------------------------------
    # 5. Generator Setup
    # ----------------------------------------
    generator = HuggingFaceLocalChatGenerator(
        model="HuggingFaceH4/zephyr-7b-alpha"
    )
    generator.warm_up()

    # ----------------------------------------
    # 6. Assemble the RAG Pipeline
    # ----------------------------------------
    pipe = Pipeline()
    pipe.add_component("text_embedder", text_embedder)
    pipe.add_component("retriever", retriever)
    pipe.add_component("prompt_builder",prompt_builder)
    pipe.add_component("generator", generator)

    pipe.connect("text_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder.prompt", "generator.messages")

    # After building pipeline
    print("Pipeline connections:")
    for connection in pipe.graph.edges:
        print(f"{connection[0]} → {connection[1]}")
    
    print(f"\n🕒 Pipeline built in {time.time()-start_time:.2f} seconds")

    return pipe

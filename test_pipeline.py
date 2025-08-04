# test_pipeline.py

from pathlib import Path
from haystack_pipeline.pipeline import build_pipeline

# 1. Build the pipeline
PROJECT_ROOT = Path(__file__).resolve().parent
pipe = build_pipeline(PROJECT_ROOT)

# 2. Run your test query
res = pipe.run({
    "text_embedder": {"text": "How do I treat maize leaf blight?"},
    "prompt_builder": {"question": "How do I treat maize leaf blight?"}
})
print(res["generator"]["replies"][0].text)

    
---
title: "Neural Search Engine"
description: "A semantic search engine for research papers using sentence transformers and FAISS."
tech: ["Python", "PyTorch", "FAISS", "FastAPI"]
github: "https://github.com/example/neural-search"
demo: "#"
---

Traditional keyword search struggles with synonymy and context. I built this neural search engine to find related research papers based on the **semantic meaning** of the query rather than exact keyword matches.

### Architecture Overview

1. **Embedding Model**: Uses `all-MiniLM-L6-v2` from SentenceTransformers to encode paper abstracts into 384-dimensional dense vectors.
2. **Vector Database**: Uses Meta's FAISS library for exact inner product search, scaling to 100k+ documents efficiently.
3. **API Layer**: A lightweight FastAPI backend that exposes the search endpoints.

The hardest challenge was optimizing the index construction time. By batching the embeddings on the GPU, I reduced the indexing time from 4 hours to roughly 15 minutes.

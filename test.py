import os
import sys

from RAG1.indexing import vector_store

# Inspect the raw dictionary storing all entries
for doc_id, data in vector_store.store.items():
    print(f"ID: {doc_id}")
    print(f"Text Snippet: {data['text'][:150]}...")
    print(f"Embedding Vector (first 5 values): {data['vector'][:5]}")
    print(f"Vector Dimensions: {len(data['vector'])}")
    print("-" * 50)
import torch
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    model_name_or_path = "infgrad/Jasper-Token-Compression-600M"
    model = SentenceTransformer(
        model_name_or_path,
        model_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",  # We support flash_attention_2; sdpa; eager
            "trust_remote_code": True
        },
        trust_remote_code=True,
        tokenizer_kwargs={"padding_side": "left"},
        device="cuda",
    )

    queries = [
        "What is photosynthesis?",
        "Who invented the telephone?",
    ]
    documents = [
        "Photosynthesis is the process by which green plants use sunlight, carbon dioxide, and water to produce glucose and oxygen",
        "Alexander Graham Bell is credited with inventing the first practical telephone in 1876, receiving US patent number 174,465 for his device."
    ]
    # The smaller the compression_ratio parameter, the faster the speed, but the quality will correspondingly decrease.
    # Based on our parameter settings during training and test results, we recommend a range between 0.3-0.8.
    query_embeddings = model.encode(queries, prompt_name="query", normalize_embeddings=True, compression_ratio=0.3333)
    document_embeddings = model.encode(documents, normalize_embeddings=True, compression_ratio=0.3333)

    similarity = model.similarity(query_embeddings, document_embeddings)
    print(similarity)

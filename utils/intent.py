from sentence_transformers import SentenceTransformer, util
import torch

# Load a small, offline-capable model
model = SentenceTransformer("all-MiniLM-L6-v2")  # ~80MB

# Predefined sample prompts for the "describe" intent
DESCRIBE_INTENT_SAMPLES = [
    "what do you see",
    "what's around you",
    "describe the scene",
    "what is in front of you",
    "can you tell me what's here",
    "tell me about your surroundings",
    "what are you seeing",
    "describe what you see",
    "describe what you see in front of you",
    "what are you looking at",
    "look around and describe it"
]

# Embed those prompts once
describe_embeddings = model.encode(DESCRIBE_INTENT_SAMPLES, convert_to_tensor=True)

def is_describe_intent(user_query, threshold=0.6):
    """
    Returns True if the user query semantically matches a 'describe' intent.
    """
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, describe_embeddings)
    max_score = torch.max(cosine_scores).item()

    print(f"[DEBUG] Intent similarity score: {max_score:.3f}")
    return max_score > threshold


import face_recognition
import numpy as np
import json

def get_embedding(image):
    rgb_img = image[:, :, ::-1]
    encodings = face_recognition.face_encodings(rgb_img)
    return encodings[0] if encodings else None

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def load_knowledge_base(path="knowledge_base.json"):
    with open(path, "r") as f:
        return json.load(f)

def save_knowledge_base(kb, path="knowledge_base.json"):
    with open(path, "w") as f:
        json.dump(kb, f, indent=2)

def load_and_chunk_document(file_path, chunk_size=150, overlap_size=25):
    """Load text from a file and split it into chunks."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append({'text': chunk})
    return chunks

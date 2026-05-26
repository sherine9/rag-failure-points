def chunk_text(text, chunk_size=256, overlap=32):
    """
    Split a document into fixed-size chunks with overlap.
    
    Args:
        text: raw document text
        chunk_size: number of words per chunk
        overlap: number of words shared between consecutive chunks
    
    Returns:
        list of text chunks
    
    Example:
        text = "The cat sat on the mat..."
        chunks = chunk_text(text, chunk_size=4, overlap=1)
        → ["The cat sat on", "on the mat..."]
    """
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # if we've reached the end, stop
        if end == len(words):
            break
            
        start += chunk_size - overlap
    
    return chunks


if __name__ == "__main__":
    # quick test
    sample = " ".join([f"word{i}" for i in range(600)])
    chunks = chunk_text(sample, chunk_size=256, overlap=32)
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk (first 10 words): {' '.join(chunks[0].split()[:10])}")
    print(f"Second chunk (first 10 words): {' '.join(chunks[1].split()[:10])}")
    print(f"Overlap check - last words of chunk 0: {' '.join(chunks[0].split()[-5:])}")
    print(f"Overlap check - first words of chunk 1: {' '.join(chunks[1].split()[:5])}")
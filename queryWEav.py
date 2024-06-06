import weaviate

def get_chunks(client, start_offset=0, num_chunks=100):
    """Retrieve and print document chunks from Weaviate with pagination."""
    while True:
        # GraphQL query with pagination
        query = f"""
        {{
            Get {{
                DocumentChunk(limit: {num_chunks}, offset: {start_offset}) {{
                    title
                    authors
                    text
                    index
                    doi_url
                }}
            }}
        }}
        """
        # Execute the GraphQL query
        result = client.query.raw(query)
        # Check for errors in the result
        if 'errors' in result:
            print("Error retrieving data:", result['errors'])
            break
        else:
            chunks = result['data']['Get']['DocumentChunk']
            if not chunks:
                print("No more chunks to display.")
                break
            for i, chunk in enumerate(chunks):
                print(f"Chunk {start_offset + i + 1}:")
                print("Title:", chunk['title'])
                print("Authors:", ', '.join(chunk['authors']))
                print("Text:", chunk['text'])
                print("Index:", chunk['index'])
                print("DOI URL:", chunk['doi_url'])
                print("-" * 60)
            start_offset += num_chunks

# Assuming client is already initialized
client = weaviate.Client("http://localhost:8080")
get_chunks(client)

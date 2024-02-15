
# possibly add a simple UI

def handle_query(rag_chain):
    query = ""
    while query != "exit":
        query = input("Ask a question about Acral Lentiginous Melanoma or type 'exit' to quit: ")
        response = rag_chain.invoke(query)
        print(response)



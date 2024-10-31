def handle_query(rag_chain):
    """Handle user queries in a loop."""
    while True:
        question = input("Please enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        response = rag_chain.invoke(question)
        print(f"Answer: {response}")

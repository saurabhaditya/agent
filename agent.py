# Necessary Imports
import os
import sys
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Set the OpenAI API key and model name
Open_API_Key = os.getenv("OPENAI_API_KEY")
if not Open_API_Key:
    Open_API_Key = os.getenv("Open_API_Key")
    if Open_API_Key:
        os.environ["OPENAI_API_KEY"] = Open_API_Key

MODEL = "gpt-4o-mini"

# Set the Tavily API key
Tavily_API_Key = os.getenv("TAVILY_API_KEY")
if not Tavily_API_Key:
    Tavily_API_Key = os.getenv("Tavily_API_Key")
    if Tavily_API_Key:
        os.environ["TAVILY_API_KEY"] = Tavily_API_Key

# Load the vectorstore
embeddings = OpenAIEmbeddings()
vector = FAISS.load_local(
    "./faiss_index", embeddings, allow_dangerous_deserialization=True
)

# Creating a retriever from the loaded vector store
retriever = vector.as_retriever(search_kwargs={"k": 5})

# Create retriever tool for Amazon product search
retriever_tool = create_retriever_tool(
    retriever,
    "amazon_product_search",
    "Search for information about Amazon products. For any questions related to Amazon products, this tool must be used."
)

# Create Tavily search tool
search_tool = TavilySearchResults(
    max_results=5,
    include_answer=True,
    include_raw_content=True,
)

# Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

# Create a list of tools
tools = [retriever_tool, search_tool]

# Initialize OpenAI model with streaming enabled
summary_llm = ChatOpenAI(model=MODEL, temperature=0, streaming=True)

# Enable memory optimization with ConversationSummaryMemory
summary_memory = ConversationSummaryMemory(
    llm=summary_llm,
    memory_key="chat_history",
    return_messages=True
)

# Create a ReAct agent
summary_react_agent = create_react_agent(
    llm=summary_llm,
    tools=tools,
    prompt=prompt
)

# Configure the AgentExecutor to manage reasoning steps
summary_agent_executor = AgentExecutor(
    agent=summary_react_agent,
    tools=tools,
    memory=summary_memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

# Initialize session-based chat history
session_memory = {}

def get_memory(session_id: str) -> ChatMessageHistory:
    """Fetch or create a chat history instance for a given session."""
    if session_id not in session_memory:
        session_memory[session_id] = ChatMessageHistory()
    return session_memory[session_id]

# Wrap agent with session-based chat history
agent_with_chat_history = RunnableWithMessageHistory(
    summary_agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history"
)

def chat_with_agent(user_input: str, session_id: str = "default") -> str:
    """Processes user input and maintains session-based chat history."""
    memory = get_memory(session_id)

    response = agent_with_chat_history.invoke(
        {"input": user_input, "chat_history": memory.messages},
        config={"configurable": {"session_id": session_id}}
    )

    if isinstance(response, dict) and "output" in response:
        return response["output"]
    else:
        return "Error: Unexpected response format"


# ==================== TEST SUITE ====================
def run_tests():
    """Run tests to verify agent functionality."""
    print("\n" + "="*60)
    print("RUNNING AGENT TESTS")
    print("="*60 + "\n")

    test_results = []

    # Test 1: Amazon Product Search (Vector Store)
    print("-" * 40)
    print("TEST 1: Amazon Product Search (Vector Store)")
    print("-" * 40)
    try:
        query1 = "What are some highly rated wireless earbuds on Amazon?"
        print(f"Query: {query1}\n")
        response1 = chat_with_agent(query1, session_id="test1")
        print(f"Response: {response1}\n")
        success1 = len(response1) > 50 and "error" not in response1.lower()
        test_results.append(("Amazon Product Search", success1, response1[:200] if len(response1) > 200 else response1))
        print(f"Result: {'PASSED' if success1 else 'FAILED'}\n")
    except Exception as e:
        test_results.append(("Amazon Product Search", False, str(e)))
        print(f"Result: FAILED - {e}\n")

    # Test 2: Another Vector Store Query
    print("-" * 40)
    print("TEST 2: Product Reviews Query (Vector Store)")
    print("-" * 40)
    try:
        query2 = "What do customers say about battery life in product reviews?"
        print(f"Query: {query2}\n")
        response2 = chat_with_agent(query2, session_id="test2")
        print(f"Response: {response2}\n")
        success2 = len(response2) > 50 and "error" not in response2.lower()
        test_results.append(("Product Reviews Query", success2, response2[:200] if len(response2) > 200 else response2))
        print(f"Result: {'PASSED' if success2 else 'FAILED'}\n")
    except Exception as e:
        test_results.append(("Product Reviews Query", False, str(e)))
        print(f"Result: FAILED - {e}\n")

    # Test 3: Tavily Web Search
    print("-" * 40)
    print("TEST 3: Web Search (Tavily)")
    print("-" * 40)
    try:
        query3 = "What are the latest tech news about artificial intelligence in December 2024?"
        print(f"Query: {query3}\n")
        response3 = chat_with_agent(query3, session_id="test3")
        print(f"Response: {response3}\n")
        success3 = len(response3) > 50 and "error" not in response3.lower()
        test_results.append(("Web Search (Tavily)", success3, response3[:200] if len(response3) > 200 else response3))
        print(f"Result: {'PASSED' if success3 else 'FAILED'}\n")
    except Exception as e:
        test_results.append(("Web Search (Tavily)", False, str(e)))
        print(f"Result: FAILED - {e}\n")

    # Test 4: General Knowledge (should use web search)
    print("-" * 40)
    print("TEST 4: General Knowledge Query (Web Search)")
    print("-" * 40)
    try:
        query4 = "What is the current price of Bitcoin?"
        print(f"Query: {query4}\n")
        response4 = chat_with_agent(query4, session_id="test4")
        print(f"Response: {response4}\n")
        success4 = len(response4) > 20 and "error" not in response4.lower()
        test_results.append(("General Knowledge Query", success4, response4[:200] if len(response4) > 200 else response4))
        print(f"Result: {'PASSED' if success4 else 'FAILED'}\n")
    except Exception as e:
        test_results.append(("General Knowledge Query", False, str(e)))
        print(f"Result: FAILED - {e}\n")

    # Print Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, success, _ in test_results if success)
    total = len(test_results)
    print(f"\nTotal: {passed}/{total} tests passed\n")
    for name, success, detail in test_results:
        status = "PASSED" if success else "FAILED"
        print(f"  [{status}] {name}")
    print()

    return test_results


# ==================== GRADIO UI ====================
def launch_gradio():
    """Launch the Gradio web interface."""
    import gradio as gr

    # Define function for Gradio interface
    def gradio_chat(user_input, session_id="gradio_session"):
        """Processes user input for Gradio interface."""
        return chat_with_agent(user_input, session_id)

    # Create Gradio app interface
    with gr.Blocks() as app:
        gr.Markdown("# Review Genie - Agents & ReAct Framework")
        gr.Markdown("Enter your query below and get AI-powered responses with session memory.")

        with gr.Row():
            input_box = gr.Textbox(label="Enter your query:", placeholder="Ask something...")
            output_box = gr.Textbox(label="Response:", lines=10)

        submit_button = gr.Button("Submit")
        submit_button.click(gradio_chat, inputs=input_box, outputs=output_box)

    # Launch the Gradio app
    app.launch(debug=True, share=True)


if __name__ == "__main__":
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("Open_API_Key"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    if not os.getenv("TAVILY_API_KEY") and not os.getenv("Tavily_API_Key"):
        print("ERROR: TAVILY_API_KEY environment variable not set")
        sys.exit(1)

    print("API Keys loaded successfully!")
    print(f"Using model: {MODEL}")
    print(f"Vector store loaded from: ./faiss_index")

    # Parse command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            run_tests()
        elif sys.argv[1] == "--gradio":
            launch_gradio()
        else:
            print("Usage: python agent.py [--test | --gradio]")
            print("  --test   : Run automated tests")
            print("  --gradio : Launch Gradio web interface")
    else:
        # Default: launch Gradio
        launch_gradio()

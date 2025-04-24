from langgraph.graph import StateGraph
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import time
import shutil
import chromadb

# Function to clear and recreate Chroma DB directory
def setup_chroma():
    chroma_dir = "./chroma_db2"
    if os.path.exists(chroma_dir):
        try:
            shutil.rmtree(chroma_dir)
            time.sleep(1)  # Wait for cleanup to complete
        except Exception as e:
            print(f"Warning: Could not clear Chroma DB directory: {e}")
    os.makedirs(chroma_dir, exist_ok=True)
    return chroma_dir

# Set API keys
load_dotenv('.env')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize Chroma DB with error handling
def initialize_vectorstore():
    try:
        # Setup Chroma directory
        chroma_dir = setup_chroma()
        
        # Initialize embeddings
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create initial document
        initial_docs = [Document(page_content="Initial document to create vectorstore")]
        
        # Create new Chroma client with explicit settings
        client = chromadb.PersistentClient(path=chroma_dir)
        
        # Initialize vectorstore
        vectorstore = Chroma.from_documents(
            documents=initial_docs,
            embedding=embedding_model,
            client=client,
            collection_name="research_memories"
        )
        return vectorstore
    except Exception as e:
        print(f"Error initializing vectorstore: {e}")
        raise

try:
    vectorstore = initialize_vectorstore()
except Exception as e:
    print(f"Failed to initialize vectorstore: {e}")
    exit(1)

# Conversation history storage
conversation_history = []
last_query_time = 0
CONVERSATION_TIMEOUT = 300  # 5 minutes

# Tavily search tool
search_tool = TavilySearchResults()

# Gemini LLM
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Define state schema
class State(dict):
    """State for the agent."""
    user_input: str
    convo_context: str = ""
    memory_context: str = ""
    search_results: str = ""
    search_sources: str = ""
    answer: str = ""
    final_output: str = ""

# [Rest of your code remains exactly the same from Node definitions onward...]
# Node: Add Conversation Context
def convo_context_node(state):
    global conversation_history, last_query_time
    current_time = time.time()
    
    if current_time - last_query_time > CONVERSATION_TIMEOUT:
        conversation_history = []
    
    last_query_time = current_time
    conversation_history.append({"role": "user", "content": state["user_input"]})
    
    recent_history = conversation_history[-10:]
    context = "\n".join([f"{'User' if item['role'] == 'user' else 'Assistant'}: {item['content']}" 
                        for item in recent_history])
    
    return {"convo_context": context}

# Node: Memory Retrieval
def memory_node(state):
    query = state["user_input"]
    docs = vectorstore.similarity_search(query, k=3)
    
    if docs and docs[0].page_content != "Initial document to create vectorstore":
        memory_context = "\n".join([f"Memory {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    else:
        memory_context = ""
    
    return {"memory_context": memory_context}

# Node: Web Search
def search_node(state):
    results = search_tool.invoke({"query": state["user_input"]})
    
    search_results = "\n".join([f"Result {i+1}: {res['content']}" for i, res in enumerate(results[:3])])
    
    search_sources = "\n".join(
        [f"Source {i+1}: {res.get('title', 'Untitled')} - {res.get('url', 'No URL available')}" 
         for i, res in enumerate(results[:3])]
    )
    
    return {
        "search_results": search_results,
        "search_sources": search_sources
    }

# Node: Generate Answer with Gemini
def gemini_node(state):
    query = state["user_input"]
    conversation = state.get("convo_context", "")
    memory = state.get("memory_context", "")
    search = state.get("search_results", "")
    sources = state.get("search_sources", "")

    prompt = f"""
You are a helpful conversational research assistant.

Answer the following question using the conversation context, memory from previous interactions, and search results.
Format your response in a clear, pointwise manner (using bullet points) whenever appropriate.
Include information about which source was used for each point by referencing the source number.
Your answer should be conversational but informative.

Recent Conversation:
{conversation}

Question:
{query}

{('Relevant Memory from Previous Conversations:\n' + memory) if memory else ''}

Search Results:
{search}

Sources:
{sources}

Instructions:
1. Answer in a bullet-point format when presenting multiple pieces of information
2. For each major point, mention which source it came from (e.g., "According to Source 1...")
3. Be conversational but informative
4. Present information in a logical order
"""

    response = gemini.invoke(prompt)
    
    conversation_history.append({"role": "assistant", "content": response.content})
    final_output = f"{response.content}\n\nSources:\n{sources}"
    
    return {
        "answer": response.content,
        "final_output": final_output
    }

# Node: Store Memory
def store_memory(state):
    content_to_store = f"Q: {state['user_input']}\nA: {state['answer']}"
    doc = Document(page_content=content_to_store, metadata={"timestamp": time.time()})
    vectorstore.add_documents([doc])
    return {}

# Create and configure the graph
workflow = StateGraph(State)

# Add all nodes to the graph
workflow.add_node("context_handler", convo_context_node)
workflow.add_node("memory", memory_node)
workflow.add_node("search", search_node)
workflow.add_node("gemini", gemini_node)
workflow.add_node("store", store_memory)

# Set the entry point
workflow.set_entry_point("context_handler")

# Add edges
workflow.add_edge("context_handler", "memory")
workflow.add_edge("memory", "search")
workflow.add_edge("search", "gemini")
workflow.add_edge("gemini", "store")

# Set the final node as the end node
workflow.set_finish_point("store")

# Compile the workflow
agent_executor = workflow.compile()

# Example usage
if __name__ == "__main__":
    # First query
    result1 = agent_executor.invoke({"user_input": "why is obesity increasing worldwide?"})
    print(result1["final_output"])
    
    # Follow-up query
    result2 = agent_executor.invoke({"user_input": "what are the main contributing factors?"})
    print(result2["final_output"])
    
    # Another follow-up
    result3 = agent_executor.invoke({"user_input": "which countries are most affected?"})
    print(result3["final_output"])
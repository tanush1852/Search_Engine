from langgraph.graph import StateGraph
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import time
import streamlit as st
import shutil

# Clear existing Chroma database to prevent errors
if os.path.exists("./chroma_db"):
    try:
        shutil.rmtree("./chroma_db")
    except Exception as e:
        print(f"Error clearing Chroma DB: {e}")

# Set API keys
load_dotenv('.env')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create documents list for initial vectorstore
initial_docs = [Document(page_content="Initial document to create vectorstore")]

# Initialize vectorstore with embeddings and documents
vectorstore = Chroma.from_documents(
    documents=initial_docs, 
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

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

# Node: Add Conversation Context
def convo_context_node(state):
    global conversation_history, last_query_time
    current_time = time.time()
    
    # Check if it's a new conversation based on time elapsed
    if current_time - last_query_time > CONVERSATION_TIMEOUT:
        conversation_history = []
    
    last_query_time = current_time
    
    # Add the current query to history
    conversation_history.append({"role": "user", "content": state["user_input"]})
    
    # Format conversation context (last 10 exchanges maximum)
    recent_history = conversation_history[-10:]
    context = "\n".join([f"{'User' if item['role'] == 'user' else 'Assistant'}: {item['content']}" 
                        for item in recent_history])
    
    return {"convo_context": context}

# Node: Memory Retrieval - Enhanced to better capture conversation context
def memory_node(state):
    query = state["user_input"]
    # Include conversation context in the search
    conversation_context = state.get("convo_context", "")
    combined_query = f"{query}\n\nContext:\n{conversation_context}"
    
    docs = vectorstore.similarity_search(combined_query, k=3)
    
    if docs and docs[0].page_content != "Initial document to create vectorstore":
        memory_context = "\n".join([f"Memory {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])
    else:
        memory_context = ""
    
    return {"memory_context": memory_context}

# Node: Web Search - Modified to include conversation context in search
def search_node(state):
    conversation_context = state.get("convo_context", "")
    enhanced_query = f"{state['user_input']}\n\nContext: {conversation_context}"
    
    results = search_tool.invoke({"query": enhanced_query})
    
    # Format search results
    search_results = "\n".join([f"Result {i+1}: {res['content']}" for i, res in enumerate(results[:3])])
    
    # Format sources directly
    search_sources = "\n".join(
        [f"Source {i+1}: {res.get('title', 'Untitled')} - {res.get('url', 'No URL available')}" 
         for i, res in enumerate(results[:3])]
    )
    
    return {
        "search_results": search_results,
        "search_sources": search_sources
    }

# Node: Generate Answer with Gemini - Enhanced to better use conversation context
def gemini_node(state):
    query = state["user_input"]
    conversation = state.get("convo_context", "")
    memory = state.get("memory_context", "")
    search = state.get("search_results", "")
    sources = state.get("search_sources", "")

    prompt = f"""
You are a helpful conversational research assistant. Maintain context from previous messages.

CONVERSATION HISTORY:
{conversation}

CURRENT QUESTION:
{query}

RELEVANT MEMORY FROM PAST CONVERSATIONS:
{memory}

SEARCH RESULTS:
{search}

INSTRUCTIONS:
1. Answer the question while maintaining conversation context
2. Use bullet points for multiple pieces of information
3. Reference sources (e.g., "According to Source 1...")
4. Be conversational but informative
5. If this is a follow-up question, connect it to previous discussion

YOUR RESPONSE:"""
    
    try:
        response = gemini.invoke(prompt)
        final_output = f"{response.content}\n\nSources:\n{sources}"
        
        # Add to conversation history
        conversation_history.append({"role": "assistant", "content": response.content})
        
        return {
            "answer": response.content,
            "final_output": final_output
        }
    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "final_output": f"Sorry, I encountered an error. Please try again.\nError: {str(e)}"
        }

# Node: Store Memory - Enhanced to store conversation context
def store_memory(state):
    content_to_store = f"Q: {state['user_input']}\nContext: {state.get('convo_context', '')}\nA: {state['answer']}"
    doc = Document(page_content=content_to_store, metadata={"timestamp": time.time()})
    try:
        vectorstore.add_documents([doc])
    except Exception as e:
        print(f"Error storing memory: {e}")
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

# Streamlit UI
def main():
    st.title("Research Assistant")
    st.write("Ask me anything and I'll research it for you!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask your question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.spinner("Researching..."):
            try:
                response = agent_executor.invoke({"user_input": prompt})
                answer = response["final_output"]
            except Exception as e:
                answer = f"Error: {str(e)}"
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
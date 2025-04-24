# research_system.py - Optimized for Free Gemini API Tier

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

# Initialize logging
logging.basicConfig(filename='research_system.log', level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ======================
# Free Tier Quota Limits
# ======================
class FreeTierQuotaManager:
    def __init__(self):
        self.reset_limits()
        
    def reset_limits(self):
        self.last_call_time = None
        self.daily_count = 0
        self.daily_limit = 20  # Conservative free tier limit (60/day)
        self.min_call_interval = 5.0  # 5s between calls
        self.total_tokens = 0
        self.token_limit = 10000  # ~30k tokens/day
    
    def check_quota(self, estimated_tokens=50):
        """Strict quota checks for free tier"""
        now = datetime.now()
        
        # Daily reset
        if self.last_call_time and now.date() > self.last_call_time.date():
            self.reset_limits()
            logger.info("Daily quota reset")
        
        # Hard limits
        if self.daily_count >= self.daily_limit:
            raise Exception(f"Daily limit reached ({self.daily_count}/{self.daily_limit})")
            
        if self.total_tokens + estimated_tokens > self.token_limit:
            raise Exception(f"Token limit ({self.total_tokens}/{self.token_limit})")
        
        # Rate limiting
        if self.last_call_time:
            elapsed = (now - self.last_call_time).total_seconds()
            if elapsed < self.min_call_interval:
                wait_time = self.min_call_interval - elapsed
                logger.info(f"Waiting {wait_time:.1f}s")
                time.sleep(wait_time)
        
        self.last_call_time = now
        self.daily_count += 1
        self.total_tokens += estimated_tokens
        logger.info(f"Quota: Calls {self.daily_count}, Tokens {self.total_tokens}")

# Initialize quota manager
quota_manager = FreeTierQuotaManager()

# ======================
# Model Configuration
# ======================
gemini_flash = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2,
    max_retries=1,
    retry_min_seconds=30,
    retry_max_seconds=60
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize Vector Database
vector_db = Chroma(embedding_function=embeddings, collection_name="research_data")

# Search Configuration
search_tool = TavilySearchResults(k=1, max_tokens=1000)  # Minimal search results

# ======================
# State Definition
# ======================
@dataclass
class AgentState:
    query: str
    research_plan: List[str] = field(default_factory=list)
    search_results: List[Dict] = field(default_factory=list)
    processed_data: Dict = field(default_factory=dict)
    verified_facts: List[Dict] = field(default_factory=dict)
    draft: str = ""
    final_answer: str = ""

# ======================
# Optimized Agents
# ======================
def controller_agent(state: AgentState) -> Dict:
    """Simplified workflow controller"""
    try:
        if not state.research_plan:
            return {"next": "research_planner"}
        elif not state.search_results:
            return {"next": "research_agent"}
        elif not state.processed_data:
            return {"next": "processing_agent"}
        else:
            state.final_answer = generate_final_output(state)
            return {"next": END}
    except Exception as e:
        logger.error(f"Controller error: {str(e)}")
        state.final_answer = f"Research error: {str(e)}"
        return {"next": END}

def research_planner(state: AgentState) -> Dict:
    """Generate 2 research questions max"""
    prompt = ChatPromptTemplate.from_template(
        "Generate 2 research questions about: {query}. Respond ONLY as JSON list."
    )
    
    try:
        chain = prompt | gemini_flash | JsonOutputParser()
        state.research_plan = safe_invoke(
            chain,
            {"query": state.query[:150]},
            agent_name="planner",
            max_tokens=100
        )[:2]  # Force limit
    except:
        state.research_plan = [
            f"What is the current state of {state.query}?",
            f"What are key challenges in {state.query}?"
        ]
    
    return {"next": "controller_agent"}

def research_agent(state: AgentState) -> Dict:
    """Perform limited search operations"""
    results = []
    
    for question in state.research_plan[:2]:  # Max 2 questions
        try:
            quota_manager.check_quota(estimated_tokens=50)
            search_result = search_tool.invoke(question)[0]
            results.append({
                "question": question,
                "content": search_result.get("content", "")[:250],
                "source": search_result.get("source", "")
            })
        except Exception as e:
            logger.warning(f"Search failed: {str(e)}")
            continue
    
    state.search_results = results
    return {"next": "controller_agent"}

def processing_agent(state: AgentState) -> Dict:
    """Consolidate and process data"""
    if not state.search_results:
        state.processed_data = {"summary": "No results found"}
        return {"next": "controller_agent"}
    
    try:
        prompt = ChatPromptTemplate.from_template(
            "Summarize this in 3 bullet points:\n{data}\nRespond with JSON: {{'summary': '...'}}"
        )
        
        chain = prompt | gemini_flash | JsonOutputParser()
        content = "\n".join([r["content"] for r in state.search_results])
        
        state.processed_data = safe_invoke(
            chain,
            {"data": content[:1000]},
            agent_name="processor",
            max_tokens=200
        )
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        state.processed_data = {"summary": "Data processing error"}
    
    return {"next": "controller_agent"}

def generate_final_output(state: AgentState) -> str:
    """Generate final answer with fallbacks"""
    try:
        prompt = ChatPromptTemplate.from_template(
            "Create 150-word answer about {query} using: {data}"
        )
        
        chain = prompt | gemini_flash
        response = safe_invoke(
            chain,
            {
                "query": state.query,
                "data": state.processed_data.get("summary", "")
            },
            max_tokens=300
        )
        
        return response.content[:500]  # Hard length limit
    except:
        return "Could not generate final answer due to system limits"

# ======================
# Core Utilities
# ======================
def safe_invoke(chain, input_data, max_tokens=50, agent_name=""):
    """Protected LLM invocation"""
    try:
        quota_manager.check_quota(estimated_tokens=max_tokens)
        logger.info(f"Invoking {agent_name}")
        
        # Input sanitization
        if isinstance(input_data, dict):
            for k in input_data:
                if isinstance(input_data[k], str):
                    input_data[k] = input_data[k][:500]
        
        start = time.time()
        result = chain.invoke(input_data)
        logger.info(f"{agent_name} completed in {time.time()-start:.1f}s")
        
        return result
    except Exception as e:
        logger.error(f"{agent_name} failed: {str(e)}")
        if "quota" in str(e).lower():
            time.sleep(60)  # Extended backoff
        raise

# ======================
# Workflow Setup
# ======================
def build_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("controller_agent", controller_agent)
    workflow.add_node("research_planner", research_planner)
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("processing_agent", processing_agent)
    
    workflow.add_edge("controller_agent", "research_planner")
    workflow.add_edge("research_planner", "controller_agent")
    workflow.add_edge("controller_agent", "research_agent")
    workflow.add_edge("research_agent", "controller_agent")
    workflow.add_edge("controller_agent", "processing_agent")
    workflow.add_edge("processing_agent", "controller_agent")
    
    workflow.set_entry_point("controller_agent")
    return workflow.compile()

# ======================
# Execution
# ======================
def run_research(query):
    """Run with strict free tier limits"""
    state = AgentState(query=query[:100])
    
    try:
        system = build_workflow()
        result = None
        
        for step in system.stream(state):
            if END in step:
                result = step[END]
                break
            
            # Enforce step limits
            if quota_manager.daily_count >= 55:  # Leave 5-call buffer
                raise Exception("Approaching daily limit")
            
            time.sleep(5)  # Conservative pacing
        
        return {
            "answer": result.final_answer,
            "sources": [r["source"] for r in result.search_results],
            "calls_used": quota_manager.daily_count,
            "tokens_used": quota_manager.total_tokens
        }
    except Exception as e:
        return {
            "answer": f"Research incomplete: {str(e)}",
            "error": True,
            "calls_used": quota_manager.daily_count
        }

if __name__ == "__main__":
    try:
        query = "What are the impacts of quantum computing on cybersecurity?"
        print(f"Starting research: {query}")
        
        result = run_research(query)
        
        print("\nResearch Results:")
        print(f"API Calls Used: {result['calls_used']}/60")
        print(f"Final Answer:\n{result['answer']}")
        
        with open("research_output.json", "w") as f:
            json.dump(result, f)
            
    except Exception as e:
        print(f"System error: {str(e)}")
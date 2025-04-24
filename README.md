# Conversational Research Assistant

A stateful AI research assistant that combines conversation memory, web search, and generative AI to provide informed responses with source attribution.

## Features

- **Conversation Memory**: Maintains context across multiple interactions
- **Knowledge Retention**: Stores and recalls past Q&A pairs in a vector database
- **Web Research**: Integrates Tavily search for up-to-date information
- **Structured Responses**: Generates bullet-point answers with proper source attribution
- **Timeout Handling**: Automatically clears stale conversations

## Architecture

```mermaid
graph TD
    A[User Input] --> B[Context Handler]
    B --> C[Memory Retrieval]
    C --> D[Web Search]
    D --> E[Answer Generation]
    E --> F[Memory Storage]
    F --> G[Formatted Output]

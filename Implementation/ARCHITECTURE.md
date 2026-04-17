# Project Name: Graph RAG vs PageRank - Thesis Educational Web App

## 1. Role and Objective
You are an expert AI & Full-Stack Developer. Your goal is to build a web application for my Bachelor Thesis that compares classical PageRank (Authority) with Vector Search / RAG (Semantic Relevance), and proposes a Hybrid Model (Graph RAG). 

## 2. Tech Stack
- **Backend**: Python with FastAPI.
- **Math/AI**: `numpy` (for PageRank/Markov Chains) and `sentence-transformers` (for generating text embeddings and calculating Cosine Similarity).
- **LLM Integration**: A mock or free API integration to pass the retrieved context to an LLM for the final answer.
- **Frontend**: Vanilla HTML/CSS/JS. No React/Vue.
- **Graph Visualization**: `Cytoscape.js` (for rendering nodes and edges dynamically).

## 3. Core Logic & Algorithms
The system processes a graph of "Research Papers". Each Node has a "Title" and an "Abstract" (content). Edges represent Citations (Links).

When a user submits a query, the backend must calculate 3 metrics:
1. **PageRank Score**: Use Power Iteration (NumPy) based on the Adjacency Matrix (edges). Apply Min-Max Normalization (0 to 1).
2. **RAG / Semantic Score**: Convert the query and all node abstracts to embeddings. Calculate Cosine Similarity. Apply Min-Max Normalization (0 to 1).
3. **Hybrid Score**: `(Weight_A * Normalized_PageRank) + (Weight_B * Normalized_RAG)`.

## 4. The UI/UX Requirements
- **Left Side**: A Cytoscape canvas displaying the nodes. 
- **Right Side**: A Search Bar and a Radio Button group to select the active algorithm ("PageRank Only", "RAG Only", "Hybrid Graph RAG").
- **Dynamic Styling**: When an algorithm is selected, the Cytoscape nodes must dynamically change size/color based on their respective score (e.g., node radius = score * 100).
- **LLM Result Box**: Below the graph, show the final LLM generated text. The prompt to the LLM should be: "Answer the user's query strictly based on this text:[Abstract of the highest scoring node]".

## 5. Development Steps
Please read this architecture carefully. Do not write the whole app at once. 
Let's start with STEP 1: Create the FastAPI backend structure, install dependencies, and write the Python algorithm for PageRank and Cosine Similarity.
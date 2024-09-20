# README.md

## Overview
This repository contains a LangChain-powered question-answering system that leverages RAG (Retrieval-Augmented Generation) techniques to respond to user queries about video quality of experience (QoE), based on various input parameters like bitrate, round-trip time (RTT), client, and server data.

Key steps in the pipeline include:
- Extracting structured information from the user's input.
- Identifying and comparing similar questions in the template (Chain of Thought approach).
- Retrieving data about video performance (bitrate, RTT), processing it, and calculating QoE.
- Using LangChain's function calling capabilities to route the response through various tools.

## Instructions for Running the Code

### 1. Install Requirements
Create and activate a virtual environment, then install the necessary dependencies by running:
```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Set the OpenAI API Key
Make sure to set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your_openai_api_key"
```

### 3. Run the Chainlit App
To run the Chainlit-based app, use the following command:
```bash
chainlit run main.py -w
```

## Explanation of Key Components

### LangChain Steps
- **Prompt Template**: The system uses prompt templates for each question to extract the relevant information (e.g., client, server, QoE).
- **Embedding and Similarity**: The user query is embedded and compared with a list of pre-defined question templates using cosine similarity to find the most relevant one.
- **Tool Calling**: Based on the user's query, the system dynamically calls the appropriate tool to fetch and calculate the required data (e.g., QoE, bitrate, RTT).
- **Final Response**: Once the tools have been executed, the results are processed into a final response, which can be translated into the user's preferred language.

### Chain of Thoughts (Routing)
The model identifies which question template most closely matches the user query and routes it through an appropriate function, allowing step-by-step execution. This modular approach makes it easier to add or modify question templates.

### RAG (Retrieval-Augmented Generation)
The model retrieves relevant data about video QoE using pre-processed bitrate and RTT information. After retrieving the data, it performs calculations to provide insights, such as which client has the worst QoE or how RTT increase affects performance.

### Tools
- **get_question1, get_question2, get_question3, get_question4**: Each of these functions performs a specific operation on the data, such as calculating the worst-performing client or finding the most consistent server.
- **Normalization**: Bitrate and RTT values are normalized before being used to calculate QoE.

### Final Thoughts
By combining LangChain's prompt management, embeddings, and function calling features with Chainlit's interactive interface, this app provides a robust system for querying video QoE metrics with language support.

# RAG System Repository

## Overview

This repository contains a RAG system that allows users to upload PDF documents, create chunks, and retrieve relevant information based on queries. The project is divided into 2 parts: 
->The first focuses on the Colab notebook and documentation.
->The second includes the web application code and deployment setup.

## Table of Contents

- [Part 1: Colab Notebook & Documentation](#part-1-documentation)
- [Part 2: Web Application & Dockerfile](#part-2-web-application)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [License](#license)

## Part 1: Documentation

- **Colab Notebook**: A Jupyter notebook demonstrating the functionality and implementation of the RAG system.
- **RAG System PDF**: Comprehensive documentation covering setup instructions, use cases, and explanations of the algorithms.

## Part 2: Web Application

- **app.py**: The Streamlit web application code that allows users to upload PDFs, create chunks, and retrieve relevant documents.
- **requirements.txt**: A list of libraries and dependencies needed to run the application.
- **Dockerfile**: Instructions for building a Docker image of the application.

## Setup Instructions

### Prerequisites

- Python 3.11.7
- Docker (if using Docker)

### For Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repository.git
   cd your_repository
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have your OpenAI API key in a file named `openai.key`.

### For Docker Setup

1. Build the Docker image:
   ```bash
   docker build -t streamlit_rag_app .
   ```

2. Run the Docker container:
   ```bash
   docker run -d -p 8501:8501 --env OPENAI_API_KEY=$(cat openai.key) streamlit_rag_app
   ```

## Usage

1. Open your web browser and navigate to `http://localhost:8501`.
2. Upload a PDF file using the provided interface.
3. Enter your query to retrieve relevant documents.

---

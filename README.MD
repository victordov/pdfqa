# PDF QA App

This is a Python application that allows you to upload a PDF file and ask questions about its content. The application uses OpenAI's GPT-3 model to understand and answer your questions.

## Tech Stack

- Python
- Streamlit for the user interface
- PyPDF2 for parsing PDF files
- OpenAI for natural language processing
- FAISS for efficient similarity search
- Langchain for text splitting and question answering

## Installation

1. Clone the repository
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Configuration

This application requires an OpenAI API key for natural language processing. Follow the steps below to add your OpenAI API key:

1. Create a file named `.env` in the root directory of the project (if it doesn't already exist).
2. Open the `.env` file in a text editor.
3. Add the following line to the `.env` file, replacing `your-openai-key` with your actual OpenAI API key:

```dotenv
OPENAI_API_KEY=your-openai-key
```

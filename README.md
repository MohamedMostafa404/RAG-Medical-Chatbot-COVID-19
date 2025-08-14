# COVID-19 Medical RAG Chatbot 🩺

This is an AI-powered Medical Chatbot designed to provide reliable information about **COVID-19**.  
It uses a **RAG (Retrieval-Augmented Generation)** approach with a PDF knowledge base containing COVID-19 information.

---

## Features

- Answer questions about COVID-19 symptoms, prevention, treatment, and vaccination.
- Powered by AI with a pre-loaded PDF for accurate responses.
- Web interface with a modern, responsive chat UI.
- Easy to run locally with FastAPI.

---

## Project Structure

RAG-Chatbot/

├─ app.py # FastAPI backend

├─ templates/

   └─ index.html # Frontend HTML

├─ static/

   └─ styles.css # CSS for UI

├─ data/

   └─ covid_info.pdf # PDF knowledge base

├─ README.md # Project documentation

├─ requirements.txt # Python dependencies


---

## Getting Started

### Prerequisites

- Python 3.11+
- Install dependencies:

```bash
pip install -r requirements.txt

```
Running Locally

```bash
uvicorn app:app --reload

```
Open your browser at: http://127.0.0.1:8000

---

Usage

Ask questions about COVID-19 in the chat interface.

The AI will answer based on the PDF knowledge base.

---

Contributing

Feel free to fork this repository, make changes, and submit pull requests.
For major changes, please open an issue first to discuss what you would like to change.

---

License

This project is open-source and available under the MIT License.






















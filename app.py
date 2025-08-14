from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="COVID-19 Medical RAG Chatbot", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Global variables for RAG system
rag_chain = None
pdf_loaded = False

# Get API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY environment variable not set!")

def initialize_rag_system():
    """Initialize RAG system with medical PDF"""
    global rag_chain, pdf_loaded
    
    if not GOOGLE_API_KEY:
        print("Cannot initialize RAG system: Google API key not configured")
        return False
    
    try:
        pdf_path = "data/Publication2.pdf"
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return False
        
        print(f"Loading PDF: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        texts = [doc.page_content for doc in docs]
        
        text_splitter = RecursiveCharacterTextSplitter(
            add_start_index=True,
            chunk_size=350,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.create_documents(texts)
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY
        )
        
        chroma_dir = tempfile.mkdtemp()
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=chroma_dir,
            collection_name="medical_chunks"
        )
        retriever = db.as_retriever()
        
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Prompt for medical context
        system_prompt = (
            "You are a knowledgeable medical assistant specializing in COVID-19. "
            "Use the given context to answer questions about COVID-19, symptoms, prevention, "
            "treatments, and vaccination. "
            "If you don't know the answer, say you don't know. "
            "Keep answers concise, clear, and professional. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        pdf_loaded = True
        print("RAG system initialized successfully!")
        return True
    
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    print("🩺 Starting COVID-19 Medical RAG Chatbot...")
    if initialize_rag_system():
        print("✅ RAG system ready - AI Medical Assistant is available!")
    else:
        print("❌ RAG system failed to initialize")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    global rag_chain
    if not rag_chain:
        return {"response": "AI Medical Assistant is not available. Please try again later.", "status": "error"}
    try:
        result = rag_chain.invoke({"input": message})
        answer = result["answer"]
        return {"response": answer, "status": "success"}
    except Exception as e:
        return {"response": f"Error processing message: {str(e)}", "status": "error"}

@app.get("/api/status")
async def get_status():
    return {
        "pdf_loaded": pdf_loaded,
        "rag_ready": rag_chain is not None,
        "api_key_configured": bool(GOOGLE_API_KEY)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

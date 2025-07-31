import streamlit as st
import requests
import PyPDF2
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from typing import List, Dict, Any
import json

# Configure page settings
st.set_page_config(
    page_title="DPR to Tender Scope Generator",
    page_icon="📋",
    layout="wide"
)

class DPRToTenderConverter:
    def __init__(self, gemini_api_key: str, openai_api_key: str = ""):
        """Initialize the converter with API keys"""
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self.use_openai_embeddings = bool(openai_api_key)
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove extra whitespace and line breaks
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic cleaning)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be page numbers or headers
            if len(line) > 10 and not re.match(r'^\d+$', line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 50:  # Only add non-empty chunks
                chunks.append(chunk.strip())
        
        return chunks
    
    def get_embedding_openai(self, text: str) -> List[float]:
        """Get embedding using OpenAI API"""
        try:
            url = "https://api.openai.com/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "input": text,
                "model": "text-embedding-3-large"
            }
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                return result['data'][0]['embedding']
            else:
                st.warning(f"OpenAI API error: {response.status_code}")
                return self.get_fallback_embedding(text)
        except Exception as e:
            st.warning(f"OpenAI embedding error: {str(e)}")
            return self.get_fallback_embedding(text)
    
    def get_embedding_gemini(self, text: str) -> List[float]:
        """Get embedding using Gemini API (REST API)"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.gemini_api_key}"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": text}]},
                "taskType": "RETRIEVAL_DOCUMENT"
            }
            
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                return result['embedding']['values']
            else:
                st.warning(f"Gemini embedding API error: {response.status_code}")
                return self.get_fallback_embedding(text)
        except Exception as e:
            st.warning(f"Gemini embedding error: {str(e)}")
            return self.get_fallback_embedding(text)
    
    def get_fallback_embedding(self, text: str) -> List[float]:
        """Fallback embedding based on text characteristics"""
        # Create a more sophisticated hash-based embedding
        words = text.lower().split()
        embedding = []
        
        # Use word statistics to create embedding
        for i in range(768):
            if i < len(words):
                word_hash = hash(words[i]) % 1000
            else:
                word_hash = hash(text[i % len(text)]) % 1000
            
            embedding.append((word_hash + i * len(text)) % 1000 / 1000.0)
        
        return embedding
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using available method"""
        if self.use_openai_embeddings:
            return self.get_embedding_openai(text)
        else:
            return self.get_embedding_gemini(text)
    
    def create_vector_database(self, chunks: List[str]) -> Dict[str, Any]:
        """Create a vector database from chunks using embeddings"""
        embeddings = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            embeddings.append(embedding)
            progress_bar.progress((i + 1) / len(chunks))
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        return {
            'chunks': chunks,
            'embeddings': np.array(embeddings)
        }
    
    def retrieve_relevant_chunks(self, query: str, vector_db: Dict[str, Any], top_k: int = 5) -> List[str]:
        """Retrieve most relevant chunks for a query using embeddings"""
        # Get query embedding (use RETRIEVAL_QUERY task type for Gemini)
        if self.use_openai_embeddings:
            query_embedding = np.array(self.get_embedding_openai(query)).reshape(1, -1)
        else:
            # For Gemini, use RETRIEVAL_QUERY task type
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.gemini_api_key}"
                headers = {"Content-Type": "application/json"}
                data = {
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": query}]},
                    "taskType": "RETRIEVAL_QUERY"
                }
                
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    result = response.json()
                    query_embedding = np.array(result['embedding']['values']).reshape(1, -1)
                else:
                    query_embedding = np.array(self.get_fallback_embedding(query)).reshape(1, -1)
            except:
                query_embedding = np.array(self.get_fallback_embedding(query)).reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, vector_db['embeddings'])[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [vector_db['chunks'][i] for i in top_indices]
    
    def generate_scope_of_work(self, relevant_chunks: List[str]) -> str:
        """Generate Scope of Work using Gemini REST API"""
        
        # Combine relevant chunks
        context = "\n\n".join(relevant_chunks)
        
        prompt = f"""
You are an expert in preparing Tender Documents from Detailed Project Reports (DPR).

You are given extracted text chunks from a DPR that are relevant to the Scope of Work.

Your task:
1. Consolidate all relevant information into a single coherent Scope of Work
2. Organize it into sections: Overview, Objectives, Deliverables, Work Breakdown, Timeline, Exclusions, Standards
3. Use formal tender language suitable for a government infrastructure project
4. Remove irrelevant details or repeated points
5. Ensure that the final text can be directly inserted into a tender document
6. Give the Timeline with correct expected number as per your Knowledge ,do not simply write x,y,z or to be specified 

Retrieved text:
{context}

Generate a comprehensive Scope of Work document:
"""
        
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.gemini_api_key}"
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 8192
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                st.error(f"Gemini API error: {response.status_code} - {response.text}")
                return "Error generating content. Please check your API key and try again."
                
        except Exception as e:
            st.error(f"Error generating Scope of Work: {str(e)}")
            return "Error generating content. Please check your API key and try again."

def main():
    st.title("📋 DPR to Tender Scope of Work Generator")
    st.markdown("Convert your Detailed Project Report (DPR) into a professional Tender Scope of Work document")
    
    # Sidebar for API keys
    with st.sidebar:
        st.header("Configuration")
        
        # Gemini API Key (Required)
        gemini_api_key = st.text_input(
            "Enter Gemini API Key", 
            type="password", 
            help="Get your API key from Google AI Studio"
        )
        
        # OpenAI API Key (Optional)
        st.subheader("Optional: Better Embeddings")
        openai_api_key = st.text_input(
            "OpenAI API Key (Optional)", 
            type="password", 
            help="For better embeddings. Leave empty to use Gemini embeddings."
        )
        
        if gemini_api_key:
            st.success("✅ Gemini API Key configured!")
            if openai_api_key:
                st.success("✅ OpenAI API Key configured!")
                st.info("Using:")
                st.write("• **Embedding**: OpenAI text-embedding-3-large")
                st.write("• **Generation**: Gemini 2.0 Flash")
            else:
                st.info("Using:")
                st.write("• **Embedding**: Gemini text-embedding-004")
                st.write("• **Generation**: Gemini 2.0 Flash")
        else:
            st.warning("Please enter your Gemini API key to continue")
    
    if not gemini_api_key:
        st.info("👈 Please enter your Gemini API key in the sidebar to get started")
        return
    
    # Initialize converter
    converter = DPRToTenderConverter(gemini_api_key, openai_api_key)
    
    # File upload
    st.header("📁 Upload DPR Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Process button
        if st.button("🚀 Generate Scope of Work", type="primary"):
            
            with st.spinner("Processing your DPR document..."):
                
                # Step 1: Extract text
                st.info("Step 1: Extracting text from PDF...")
                raw_text = converter.extract_text_from_pdf(uploaded_file)
                
                if not raw_text:
                    st.error("Failed to extract text from PDF. Please try another file.")
                    return
                
                # Step 2: Clean text
                st.info("Step 2: Cleaning extracted text...")
                cleaned_text = converter.clean_text(raw_text)
                
                # Step 3: Create chunks
                st.info("Step 3: Creating text chunks...")
                chunks = converter.chunk_text(cleaned_text)
                st.success(f"Created {len(chunks)} text chunks")
                
                # Step 4: Create vector database
                st.info("Step 4: Creating vector database...")
                vector_db = converter.create_vector_database(chunks)
                
                # Step 5: Retrieve relevant chunks
                st.info("Step 5: Retrieving relevant information...")
                
                # Define comprehensive scope of work queries based on your document
                core_keywords = [
                    "scope of work",
                    "statement of work", 
                    "project scope",
                    "deliverables",
                    "work breakdown structure",
                    "responsibilities",
                    "technical requirements",
                    "implementation plan",
                    "construction activities",
                    "installation requirements",
                    "execution methodology",
                    "bill of quantities"
                ]
                
                contextual_keywords = [
                    "project objectives",
                    "technical specifications",
                    "schedule of work",
                    "milestones",
                    "phasing plan", 
                    "construction schedule",
                    "operational requirements",
                    "design requirements",
                    "execution plan",
                    "site preparation",
                    "installation works",
                    "completion criteria",
                    "handover requirements"
                ]
                
                semantic_queries = [
                    "Extract all details describing the scope of work for this project",
                    "What tasks and deliverables are expected from the contractor?",
                    "Summarize the technical and operational requirements for execution",
                    "Describe the methodology for executing this project",
                    "List all milestones and timelines for completion",
                    "Identify all deliverables and outputs required by the contractor",
                    "Summarize all installation, testing, and commissioning requirements",
                    "Outline all responsibilities and obligations of the contractor"
                ]
                
                # Combine all query types
                all_queries = core_keywords + contextual_keywords + semantic_queries
                
                all_relevant_chunks = []
                progress_retrieval = st.progress(0)
                
                for i, query in enumerate(all_queries):
                    relevant_chunks = converter.retrieve_relevant_chunks(query, vector_db, top_k=2)
                    all_relevant_chunks.extend(relevant_chunks)
                    progress_retrieval.progress((i + 1) / len(all_queries))
                
                # Remove duplicates while preserving order
                seen = set()
                unique_chunks = []
                for chunk in all_relevant_chunks:
                    if chunk not in seen:
                        seen.add(chunk)
                        unique_chunks.append(chunk)
                
                # Step 6: Generate Scope of Work
                st.info("Step 6: Generating Scope of Work document...")
                
                # Use top chunks (limit to prevent token overflow)
                top_chunks = unique_chunks[:15]  # Increased from 10 to 15 for better coverage
                
                st.info(f"Using {len(top_chunks)} most relevant chunks for generation...")
                scope_of_work = converter.generate_scope_of_work(top_chunks)
            
            st.success("✅ Scope of Work generated successfully!")
            
            # Display results
            st.header("📄 Generated Scope of Work")
            
            # Create two columns
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(scope_of_work)
            
            with col2:
                st.subheader("Document Stats")
                st.metric("Original Text Length", f"{len(raw_text):,} chars")
                st.metric("Cleaned Text Length", f"{len(cleaned_text):,} chars")
                st.metric("Text Chunks Created", len(chunks))
                st.metric("Total Queries Processed", len(all_queries))
                st.metric("Relevant Chunks Found", len(unique_chunks))
                st.metric("Top Chunks Used", len(top_chunks))
                
                # Download button
                st.download_button(
                    label="📥 Download Scope of Work",
                    data=scope_of_work,
                    file_name="scope_of_work.txt",
                    mime="text/plain"
                )
            
            # Show retrieval statistics
            with st.expander("📊 Retrieval Query Statistics"):
                st.subheader("Query Categories Used:")
                st.write(f"**Core Keywords**: {len(core_keywords)} queries")
                st.write(f"**Contextual Keywords**: {len(contextual_keywords)} queries") 
                st.write(f"**Semantic Queries**: {len(semantic_queries)} queries")
                st.write(f"**Total Queries**: {len(all_queries)}")
                st.write(f"**Unique Chunks Retrieved**: {len(unique_chunks)}")
                
                # Show sample queries
                st.subheader("Sample Queries Used:")
                sample_queries = [
                    "Core: " + core_keywords[0],
                    "Contextual: " + contextual_keywords[0], 
                    "Semantic: " + semantic_queries[0]
                ]
                for query in sample_queries:
                    st.write(f"• {query}")
            
            # Show preview of extracted text
            with st.expander("📖 Preview Extracted Text (First 1000 characters)"):
                st.text(cleaned_text[:1000] + "..." if len(cleaned_text) > 1000 else cleaned_text)
            
            # Show chunks used
            with st.expander("🔍 Top Relevant Chunks Used"):
                for i, chunk in enumerate(top_chunks[:5], 1):
                    st.subheader(f"Chunk {i}")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)

if __name__ == "__main__":
    main()

import streamlit as st
import google.generativeai as genai
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
    def __init__(self, api_key: str):
        """Initialize the converter with Gemini API key"""
        genai.configure(api_key=api_key)
        self.generation_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # For embeddings, we'll use text-embedding-004 model
        self.embedding_model_name = "text-embedding-004"
        
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
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Gemini embedding model"""
        try:
            # Use the embed_content function from google.generativeai
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
            
        except Exception as e:
            st.warning(f"Error generating embedding, using fallback: {str(e)}")
            # Fallback: create a hash-based embedding
            text_hash = hash(text)
            embedding = []
            for i in range(768):  # Standard embedding size
                embedding.append((text_hash * (i + 1)) % 1000 / 1000.0)
            return embedding
    
    def create_vector_database(self, chunks: List[str]) -> Dict[str, Any]:
        """Create a vector database from chunks using embeddings"""
        embeddings = []
        progress_bar = st.progress(0)
        
        for i, chunk in enumerate(chunks):
            try:
                # Get embedding for each chunk
                result = genai.embed_content(
                    model=self.embedding_model_name,
                    content=chunk,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
                
            except Exception as e:
                st.warning(f"Error with chunk {i+1}, using fallback embedding")
                # Use fallback embedding
                embedding = self.get_embedding(chunk)
                embeddings.append(embedding)
            
            progress_bar.progress((i + 1) / len(chunks))
        
        return {
            'chunks': chunks,
            'embeddings': np.array(embeddings)
        }
    
    def retrieve_relevant_chunks(self, query: str, vector_db: Dict[str, Any], top_k: int = 5) -> List[str]:
        """Retrieve most relevant chunks for a query using proper embeddings"""
        try:
            # Get query embedding
            result = genai.embed_content(
                model=self.embedding_model_name,
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = np.array(result['embedding']).reshape(1, -1)
            
        except Exception as e:
            st.warning(f"Using fallback embedding for query: {str(e)}")
            query_embedding = np.array(self.get_embedding(query)).reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, vector_db['embeddings'])[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [vector_db['chunks'][i] for i in top_indices]
    
    def generate_scope_of_work(self, relevant_chunks: List[str]) -> str:
        """Generate Scope of Work using Gemini"""
        
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

Retrieved text:
{context}

Generate a comprehensive Scope of Work document:
"""
        
        try:
            response = self.generation_model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating Scope of Work: {str(e)}")
            return "Error generating content. Please check your API key and try again."

def main():
    st.title("📋 DPR to Tender Scope of Work Generator")
    st.markdown("Convert your Detailed Project Report (DPR) into a professional Tender Scope of Work document")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password", help="Get your API key from Google AI Studio")
        
        if api_key:
            st.success("API Key configured!")
            st.info("Using Models:")
            st.write("• **Embedding**: text-embedding-004")
            st.write("• **Generation**: gemini-2.0-flash-exp")
        else:
            st.warning("Please enter your Gemini API key to continue")
    
    if not api_key:
        st.info("👈 Please enter your Gemini API key in the sidebar to get started")
        return
    
    # Initialize converter
    converter = DPRToTenderConverter(api_key)
    
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
                
                # Define scope of work queries
                sow_queries = [
                    "scope of work deliverables and responsibilities",
                    "technical requirements and specifications",
                    "project objectives and implementation plan",
                    "construction activities and installation requirements",
                    "milestones timeline and completion criteria"
                ]
                
                all_relevant_chunks = []
                for query in sow_queries:
                    relevant_chunks = converter.retrieve_relevant_chunks(query, vector_db, top_k=3)
                    all_relevant_chunks.extend(relevant_chunks)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_chunks = []
                for chunk in all_relevant_chunks:
                    if chunk not in seen:
                        seen.add(chunk)
                        unique_chunks.append(chunk)
                
                # Step 6: Generate Scope of Work
                st.info("Step 6: Generating Scope of Work document...")
                scope_of_work = converter.generate_scope_of_work(unique_chunks[:10])  # Limit to top 10 chunks
            
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
                st.metric("Relevant Chunks Used", len(unique_chunks[:10]))
                
                # Download button
                st.download_button(
                    label="📥 Download Scope of Work",
                    data=scope_of_work,
                    file_name="scope_of_work.txt",
                    mime="text/plain"
                )
            
            # Show preview of extracted text
            with st.expander("📖 Preview Extracted Text (First 1000 characters)"):
                st.text(cleaned_text[:1000] + "..." if len(cleaned_text) > 1000 else cleaned_text)
            
            # Show chunks used
            with st.expander("🔍 Relevant Chunks Used"):
                for i, chunk in enumerate(unique_chunks[:5], 1):
                    st.subheader(f"Chunk {i}")
                    st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)

if __name__ == "__main__":
    main()
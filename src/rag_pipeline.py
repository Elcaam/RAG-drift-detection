"""
RAG Pipeline implementation with comprehensive drift detection capabilities.

This module provides a complete Retrieval-Augmented Generation (RAG) pipeline
that can monitor performance drift over time as knowledge bases evolve.

Key Features:
- Document processing and chunking with configurable parameters
- FAISS vector store for efficient similarity search
- Sentence transformer embeddings for semantic understanding
- Knowledge base updates and management
- Performance evaluation and metrics computation
- Comprehensive result tracking for drift analysis

Author: Cam
Date: 2025-05-16
"""

import os
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm


class RAGPipeline:
    """
    Comprehensive RAG pipeline with drift detection capabilities.
    
    This class implements a complete Retrieval-Augmented Generation system that:
    - Processes documents and creates vector embeddings
    - Performs similarity-based retrieval
    - Evaluates performance metrics for drift detection
    - Manages knowledge base updates
    - Provides comprehensive result tracking
    """
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 top_k: int = 5):
        """
        Initialize the RAG pipeline with configurable parameters.
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Number of characters to overlap between chunks
            top_k: Number of top documents to retrieve for each query
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize components
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.vectorstore = None
        self.knowledge_base = []
        
    def create_documents(self, data: pd.DataFrame) -> List[Document]:
        """
        Create LangChain Document objects from the input dataset.
        
        Args:
            data: Input dataset with columns: 'id', 'title', 'context', 
                  'question', 'answer_text', 'answer_start'
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for _, row in data.iterrows():
            # Create metadata for tracking
            metadata = {
                'id': row['id'],
                'title': row['title'],
                'question': row['question'],
                'answer': row['answer_text'],
                'answer_start': row['answer_start']
            }
            
            # Create document with context as content
            doc = Document(
                page_content=row['context'],
                metadata=metadata
            )
            documents.append(doc)
            
        return documents
    
    def build_vectorstore(self, data: pd.DataFrame) -> None:
        """
        Build the FAISS vector store from the input dataset.
        
        Args:
            data: Input dataset containing documents to be indexed
        """
        print("Creating documents...")
        documents = self.create_documents(data)
        
        print("Building vector store...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Store knowledge base for reference
        self.knowledge_base = data.to_dict('records')
        
        print(f"Vector store built with {len(documents)} documents")
    
    def retrieve_passages(self, query: str) -> List[Dict]:
        """
        Retrieve relevant passages for a given query.
        
        Args:
            query: Input question
            
        Returns:
            List of retrieved passages with metadata
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not built. Call build_vectorstore() first.")
        
        # Retrieve documents using similarity search
        docs = self.vectorstore.similarity_search(query, k=self.top_k)
        
        # Format results
        passages = []
        for doc in docs:
            passages.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': None  # Will be computed separately if needed
            })
        
        return passages
    
    def compute_similarity_scores(self, query: str, passages: List[Dict]) -> List[float]:
        """
        Compute cosine similarity scores between query and passages.
        
        Args:
            query: Input question
            passages: List of retrieved passages
            
        Returns:
            List of similarity scores
        """
        # Encode query and passages
        query_embedding = self.embeddings.embed_query(query)
        passage_embeddings = self.embeddings.embed_documents([p['content'] for p in passages])
        
        # Compute cosine similarities
        similarities = []
        for passage_emb in passage_embeddings:
            similarity = np.dot(query_embedding, passage_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(passage_emb)
            )
            similarities.append(similarity)
        
        return similarities
    
    def answer_question(self, query: str, ground_truth: str = None) -> Dict:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            query: Input question
            ground_truth: Ground truth answer for evaluation
            
        Returns:
            Dictionary with answer and metadata
        """
        # Retrieve passages
        passages = self.retrieve_passages(query)
        
        # Compute similarity scores
        similarities = self.compute_similarity_scores(query, passages)
        
        # Add scores to passages
        for i, passage in enumerate(passages):
            passage['score'] = similarities[i]
        
        # Return the most relevant passage's answer
        # In a full implementation, you would use a generator model here
        best_passage = max(passages, key=lambda x: x['score'])
        
        result = {
            'query': query,
            'answer': best_passage['metadata']['answer'],
            'passages': passages,
            'ground_truth': ground_truth,
            'best_passage_score': best_passage['score']
        }
        
        return result
    
    def evaluate_retrieval(self, test_data: pd.DataFrame) -> Dict:
        """
        Evaluate retrieval performance on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Evaluating retrieval performance...")
        
        similarities = []
        recall_at_k = []
        
        for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
            query = row['question']
            ground_truth = row['answer_text']
            
            # Get answer
            result = self.answer_question(query, ground_truth)
            
            # Store similarity score
            similarities.append(result['best_passage_score'])
            
            # Check if ground truth is in retrieved passages
            retrieved_answers = [p['metadata']['answer'] for p in result['passages']]
            if ground_truth in retrieved_answers:
                recall_at_k.append(1)
            else:
                recall_at_k.append(0)
        
        metrics = {
            'avg_similarity': np.mean(similarities),
            'std_similarity': np.std(similarities),
            'recall_at_k': np.mean(recall_at_k),
            'num_queries': len(test_data)
        }
        
        return metrics
    
    def save_pipeline(self, filepath: str) -> None:
        """Save the pipeline to disk."""
        if self.vectorstore is not None:
            self.vectorstore.save_local(filepath)
            print(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> None:
        """Load the pipeline from disk."""
        if os.path.exists(filepath):
            self.vectorstore = FAISS.load_local(filepath, self.embeddings)
            print(f"Pipeline loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
    
    def update_knowledge_base(self, new_data: pd.DataFrame) -> None:
        """
        Update the knowledge base with new data.
        
        Args:
            new_data: New data to add to the knowledge base
        """
        print("Updating knowledge base...")
        
        # Create new documents
        new_documents = self.create_documents(new_data)
        
        # Add to existing vector store
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(new_documents, self.embeddings)
        else:
            self.vectorstore.add_documents(new_documents)
        
        # Update knowledge base
        new_kb = new_data.to_dict('records')
        self.knowledge_base.extend(new_kb)
        
        print(f"Added {len(new_documents)} new documents to knowledge base")


class SimpleGenerator:
    """Simple answer generator for demonstration purposes."""
    
    def __init__(self):
        """Initialize the generator."""
        pass
    
    def generate_answer(self, query: str, passages: List[Dict]) -> str:
        """
        Generate an answer based on retrieved passages.
        
        Args:
            query: Input question
            passages: Retrieved passages
            
        Returns:
            Generated answer
        """
        # For simplicity, return the answer from the most relevant passage
        # In a real implementation, you would use a language model here
        best_passage = max(passages, key=lambda x: x['score'])
        return best_passage['metadata']['answer']


if __name__ == "__main__":
    # Example usage
    from data_loader import SquadDataLoader
    
    # Load data
    loader = SquadDataLoader()
    train_data, test_data = loader.split_train_test(max_samples=1000)
    
    # Build pipeline
    pipeline = RAGPipeline()
    pipeline.build_vectorstore(train_data)
    
    # Evaluate
    metrics = pipeline.evaluate_retrieval(test_data[:100])
    print("Retrieval Metrics:", metrics)
    
    # Save pipeline
    pipeline.save_pipeline("data/rag_pipeline")

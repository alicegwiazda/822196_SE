"""
Unit tests for the Art Guide application. Tests individual components in isolation.
"""

import unittest
import os
import sys
import numpy as np
import torch
from PIL import Image
import tempfile
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import embed_image, generate_description


class TestEmbeddingGeneration(unittest.TestCase):
    """Test suite for image embedding generation."""
    
    def setUp(self):
        """Create a test image for embedding tests."""
        self.test_image = Image.new('RGB', (224, 224), color='red')
    
    def test_embedding_dimension(self):
        """Test that embeddings have the correct dimension (512 for CLIP ViT-B/32)."""
        embedding = embed_image(self.test_image)
        self.assertEqual(embedding.shape[1], 512, 
                        "Embedding should have 512 dimensions for CLIP ViT-B/32")
    
    def test_embedding_normalization(self):
        """Test that embeddings are L2-normalized."""
        embedding = embed_image(self.test_image)
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5,
                              msg="Embedding should be L2-normalized to 1.0")
    
    def test_embedding_reproducibility(self):
        """Test that the same image produces the same embedding."""
        embedding1 = embed_image(self.test_image)
        embedding2 = embed_image(self.test_image)
        np.testing.assert_array_almost_equal(
            embedding1, embedding2, decimal=5,
            err_msg="Same image should produce identical embeddings"
        )
    
    def test_embedding_type(self):
        """Test that embedding returns numpy array with float32 dtype."""
        embedding = embed_image(self.test_image)
        self.assertIsInstance(embedding, np.ndarray,
                            "Embedding should be a numpy array")
        self.assertEqual(embedding.dtype, np.float32,
                        "Embedding should be float32 type")


class TestImageUploadAndPreprocessing(unittest.TestCase):
    """Test suite for image upload and preprocessing."""
    
    def test_valid_image_formats(self):
        """Test that valid image formats (JPEG, PNG) can be loaded."""
        formats = [('JPEG', 'test.jpg'), ('PNG', 'test.png')]
        
        for fmt, filename in formats:
            with tempfile.NamedTemporaryFile(suffix=f'.{fmt.lower()}', delete=False) as tmp:
                test_img = Image.new('RGB', (100, 100), color='blue')
                test_img.save(tmp.name, format=fmt)
                
                # Test loading
                loaded_img = Image.open(tmp.name)
                self.assertIsInstance(loaded_img, Image.Image,
                                    f"Should load {fmt} images successfully")
                
                # Cleanup
                os.unlink(tmp.name)
    
    def test_image_to_embedding_pipeline(self):
        """Test the complete pipeline from image to embedding."""
        test_image = Image.new('RGB', (256, 256), color='green')
        embedding = embed_image(test_image)
        
        # Check output is valid
        self.assertIsNotNone(embedding, "Embedding should not be None")
        self.assertGreater(embedding.shape[1], 0, "Embedding should have features")
    
    def test_different_image_sizes(self):
        """Test that images of different sizes can be processed."""
        sizes = [(100, 100), (224, 224), (500, 500), (800, 600)]
        
        for size in sizes:
            test_img = Image.new('RGB', size, color='yellow')
            embedding = embed_image(test_img)
            self.assertEqual(embedding.shape[1], 512,
                           f"Image of size {size} should produce 512-dim embedding")


class TestMetadataRetrieval(unittest.TestCase):
    """Test suite for metadata handling."""
    
    def test_metadata_structure(self):
        """Test that metadata has the required columns."""
        # Create sample metadata
        metadata = pd.DataFrame({
            'artist': ['Van Gogh', 'Monet'],
            'title': ['Starry Night', 'Water Lilies'],
            'period': ['Post-Impressionism', 'Impressionism'],
            'image_path': ['path1.jpg', 'path2.jpg']
        })
        
        required_cols = ['artist', 'title', 'period', 'image_path']
        for col in required_cols:
            self.assertIn(col, metadata.columns,
                         f"Metadata should contain '{col}' column")
    
    def test_metadata_retrieval_format(self):
        """Test that metadata can be retrieved in the expected format."""
        metadata = pd.DataFrame({
            'artist': ['Picasso'],
            'title': ['Guernica'],
            'period': ['Cubism'],
            'image_path': ['guernica.jpg']
        })
        
        # Test retrieval
        result = metadata.iloc[0]
        self.assertEqual(result['artist'], 'Picasso')
        self.assertEqual(result['title'], 'Guernica')
        self.assertEqual(result['period'], 'Cubism')


class TestLLMPromptFormatting(unittest.TestCase):
    """Test suite for LLM prompt generation and description formatting."""
    
    def test_description_generation_returns_string(self):
        """Test that description generation returns a string."""
        description = generate_description("Van Gogh", "Starry Night", "Post-Impressionism")
        self.assertIsInstance(description, str,
                            "Description should be a string")
    
    def test_description_contains_metadata(self):
        """Test that generated description contains the provided metadata."""
        artist = "Claude Monet"
        title = "Impression, Sunrise"
        period = "Impressionism"
        
        description = generate_description(artist, title, period)
        
        self.assertIn(title, description,
                     "Description should contain artwork title")
        self.assertIn(artist, description,
                     "Description should contain artist name")
        self.assertIn(period, description,
                     "Description should contain period/style")
    
    def test_description_non_empty(self):
        """Test that description is not empty."""
        description = generate_description("Artist", "Title", "Period")
        self.assertGreater(len(description), 0,
                          "Description should not be empty")
    
    def test_description_handles_special_characters(self):
        """Test that description handles special characters in metadata."""
        description = generate_description(
            "Pablo Picasso", 
            "Les Demoiselles d'Avignon", 
            "Proto-Cubism"
        )
        self.assertIsInstance(description, str,
                            "Should handle special characters")


class TestVectorSearchLogic(unittest.TestCase):
    """Test suite for vector similarity search logic."""
    
    def test_similarity_score_range(self):
        """Test that similarity scores are in valid range."""
        # Create mock distances (FAISS returns L2 distances)
        distances = np.array([[0.1, 0.5, 0.9, 1.2, 1.5]])
        
        # Distances should be non-negative
        self.assertTrue(np.all(distances >= 0),
                       "All distances should be non-negative")
    
    def test_top_k_ordering(self):
        """Test that top-k results are ordered by similarity."""
        distances = np.array([[0.1, 0.5, 0.3, 0.9, 0.2]])
        indices = np.array([[0, 4, 2, 1, 3]])
        
        # Check that distances are in ascending order (smaller is better)
        sorted_distances = distances[0][indices[0]]
        self.assertTrue(np.all(sorted_distances[:-1] <= sorted_distances[1:]),
                       "Distances should be in ascending order")


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)

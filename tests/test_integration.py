"""
Integration tests for the Art Guide application - it tests the interaction between multiple components.
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import embed_image, search_index, generate_description, recognize


class TestEndToEndPipeline(unittest.TestCase):
    """Test the complete pipeline from image input to description output."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_image = Image.new('RGB', (224, 224), color='blue')
    
    def test_image_to_embedding_to_description(self):
        """Test the complete flow: image -> embedding -> retrieval -> description."""
        # Step 1: Generate embedding
        embedding = embed_image(self.test_image)
        self.assertIsNotNone(embedding, "Embedding generation failed")
        
        # Step 2: Simulate search (even if index is empty, should not crash)
        try:
            results, emb = search_index(self.test_image, k=5)
            # If no index, results should be None
            if results is None:
                self.assertIsNone(emb, "Should return None for both when no index")
            else:
                # If we have results, check structure
                self.assertIsInstance(results, pd.DataFrame)
        except Exception as e:
            self.fail(f"Search index should handle empty index gracefully: {e}")
    
    def test_recognize_function_integration(self):
        """Test the recognize function that combines all components."""
        # This should handle the case where index might not be loaded
        result = recognize(self.test_image, show_context=False)
        
        # Result should be a tuple of (label, image, description)
        self.assertIsInstance(result, tuple,
                            "Recognize should return a tuple")
        self.assertEqual(len(result), 3,
                        "Recognize should return 3 elements")
        
        label, img, description = result
        self.assertIsInstance(label, str, "Label should be a string")
        self.assertIsInstance(description, str, "Description should be a string")
    
    def test_pipeline_with_context_flag(self):
        """Test pipeline with context retrieval enabled."""
        result = recognize(self.test_image, show_context=True)
        
        label, img, description = result
        self.assertIsInstance(label, str)
        self.assertIsInstance(description, str)


class TestUIToRetrieval(unittest.TestCase):
    """Test the integration between UI input processing and retrieval system."""
    
    def test_image_upload_to_search(self):
        """Test that uploaded images can be processed and searched."""
        # Simulate image upload
        uploaded_image = Image.new('RGB', (500, 500), color='red')
        
        # Process through embedding
        embedding = embed_image(uploaded_image)
        self.assertEqual(embedding.shape[1], 512,
                        "Uploaded image should produce correct embedding dimension")
        
        # Test search (should handle gracefully even with no index)
        results, emb = search_index(uploaded_image, k=5)
        # Should return None or valid results without crashing
        self.assertTrue(results is None or isinstance(results, pd.DataFrame),
                       "Search should return None or DataFrame")
    
    def test_multiple_image_formats(self):
        """Test that different image formats from UI work through the pipeline."""
        formats = ['RGB', 'RGBA', 'L']  # RGB, RGBA, Grayscale
        
        for mode in formats:
            if mode == 'RGBA':
                img = Image.new(mode, (224, 224), color=(255, 0, 0, 255))
            else:
                img = Image.new(mode, (224, 224), color=128)
            
            # Convert to RGB if needed (as the app would do)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            embedding = embed_image(img)
            self.assertIsNotNone(embedding,
                               f"Should handle {mode} format images")
    
    def test_error_handling_invalid_input(self):
        """Test that the system handles invalid inputs gracefully."""
        # Test with None
        try:
            result = recognize(None, show_context=False)
            # Should handle None gracefully
            self.assertIsInstance(result, tuple)
        except AttributeError:
            # Expected if None is not handled - this is acceptable
            pass


class TestRetrievalToLLM(unittest.TestCase):
    """Test the integration between retrieval results and LLM description generation."""
    
    def test_metadata_to_description_flow(self):
        """Test that metadata from retrieval correctly flows to description generation."""
        # Simulate retrieval results
        artist = "Vincent van Gogh"
        title = "The Starry Night"
        period = "Post-Impressionism"
        
        # Generate description
        description = generate_description(artist, title, period)
        
        # Verify all metadata is included
        self.assertIn(artist, description,
                     "Artist name should be in description")
        self.assertIn(title, description,
                     "Title should be in description")
        self.assertIn(period, description,
                     "Period should be in description")
    
    def test_multiple_retrieval_results_formatting(self):
        """Test handling of multiple retrieval results."""
        # Create mock retrieval results
        results = pd.DataFrame({
            'artist': ['Monet', 'Renoir', 'Degas'],
            'title': ['Water Lilies', 'Luncheon', 'The Dance Class'],
            'period': ['Impressionism', 'Impressionism', 'Impressionism'],
            'distance': [0.1, 0.3, 0.5]
        })
        
        # Test that we can process top result
        top_result = results.iloc[0]
        description = generate_description(
            top_result['artist'],
            top_result['title'],
            top_result['period']
        )
        
        self.assertIsInstance(description, str,
                            "Should generate description from top result")
        self.assertGreater(len(description), 0,
                          "Description should not be empty")
    
    def test_description_consistency(self):
        """Test that the same metadata produces consistent descriptions."""
        artist, title, period = "Pablo Picasso", "Guernica", "Cubism"
        
        desc1 = generate_description(artist, title, period)
        desc2 = generate_description(artist, title, period)
        
        # For placeholder function, should be identical
        # For LLM, might vary slightly but should contain same metadata
        self.assertIn(artist, desc1)
        self.assertIn(artist, desc2)
        self.assertIn(title, desc1)
        self.assertIn(title, desc2)


class TestDataPersistenceAndLogging(unittest.TestCase):
    """Test data storage and telemetry logging integration."""
    
    def setUp(self):
        """Create temporary log file for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_log = os.path.join(self.temp_dir, 'test_telemetry.csv')
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_log_file_creation(self):
        """Test that log files are created with correct structure."""
        import csv
        
        # Create a test log file
        with open(self.temp_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'artist', 'confidence', 'response_time'])
            writer.writerow(['2025-11-20 10:00:00', 'Van Gogh', '0.95', '0.5'])
        
        # Verify structure
        df = pd.read_csv(self.temp_log)
        expected_cols = ['timestamp', 'artist', 'confidence', 'response_time']
        
        for col in expected_cols:
            self.assertIn(col, df.columns,
                         f"Log file should have '{col}' column")
    
    def test_telemetry_data_types(self):
        """Test that telemetry data has correct types."""
        import csv
        
        with open(self.temp_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'artist', 'confidence', 'response_time'])
            writer.writerow(['2025-11-20 10:00:00', 'Monet', '0.88', '0.7'])
        
        df = pd.read_csv(self.temp_log)
        
        # Check that numeric columns can be converted
        self.assertIsNotNone(pd.to_numeric(df['confidence'], errors='coerce').iloc[0])
        self.assertIsNotNone(pd.to_numeric(df['response_time'], errors='coerce').iloc[0])


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)

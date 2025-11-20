# Art Guide Project

**Team:** AlBeSa  
**Authors:** Aleksandra Nowińska, Beloslava Malakova

## Overview

This project implements an AI-powered artwork recognition system designed to enhance museum experiences. The application allows users to upload photographs of artworks and receive detailed, contextual information including artist identification, historical background, and artistic analysis. The system combines computer vision for artwork recognition with large language models for generating educational descriptions.

## Problem Definition

Museum visitors frequently encounter barriers to accessing comprehensive artwork information. Physical labels provide minimal context, audio guides are not always available, and language barriers limit accessibility for international visitors. This system addresses these challenges by providing immediate, detailed artwork analysis through an intuitive web interface that generates both visual recognition results and contextual explanations suitable for diverse audiences.

## System Architecture

The application supports two deployment configurations. The monolithic deployment runs as a single Gradio application on port 7860 and is suitable for development and demonstration purposes. This mode integrates the web interface, CLIP-based image processing, FAISS vector search, and Gemini API integration in one process.

The distributed architecture implements a three-tier design with separate services for improved scalability. The interface server handles user interactions through a Flask application on port 5000. The orchestrator service uses Redis as a message queue to coordinate asynchronous communication between components. The AI server performs computational tasks including CLIP embedding generation, FAISS similarity search, and Gemini API calls for description generation. This architecture enables horizontal scaling by supporting multiple AI server instances and provides fault tolerance through service isolation.

## Dataset

The system operates on curated artwork datasets containing 50 images across five artists with ten artworks each. The dataset includes images in JPEG format along with structured metadata containing artist names, artwork titles, creation periods, and artistic movement classifications. Images are stored in the data/artworks directory while metadata is maintained in CSV format for efficient processing.

Data preprocessing involves validating metadata consistency, ensuring proper image-metadata alignment, and normalizing image dimensions for embedding generation. The dataset remains static during operation with the FAISS index built once during initialization. Sources include publicly available museum collections and WikiArt datasets to ensure reliability and historical accuracy.

## Model Architecture

The recognition pipeline consists of three primary components working in sequence. First, the CLIP ViT-B/32 model generates 512-dimensional embeddings from input images. This pre-trained vision-language model from OpenAI provides robust feature representations without requiring additional training. Second, the FAISS library performs efficient similarity search using L2 distance metrics to identify the five most similar artworks from the indexed dataset. Third, the Gemini 2.5 Flash API generates comprehensive descriptions between 300 and 400 words based on the recognized artwork metadata.

The processing pipeline follows this sequence: user uploads an image, CLIP generates the embedding vector, FAISS searches the indexed vectors and returns top five matches with distance scores, metadata for the closest match is retrieved, and Gemini generates a structured description including introduction, artist background, artistic analysis, historical context, and cultural impact. When the Gemini API key is not configured, the system provides a detailed fallback template description ensuring continuous operation.

Performance metrics show 87% top-one accuracy and 96% top-five accuracy on the test dataset. End-to-end response time averages between eight and ten seconds including image processing, vector search, and description generation. Inference time for embedding generation is approximately one to two seconds on CPU and under half a second on GPU-enabled systems.

## Technology Stack

The application is implemented in Python 3.13 using modern machine learning and web development libraries. Core dependencies include PyTorch 2.9.1 for deep learning operations and the Transformers library for CLIP model integration. Data processing relies on pandas and numpy for efficient manipulation of metadata and numerical arrays.

The vector search functionality uses faiss-cpu for efficient similarity search across high-dimensional embeddings. Language model integration is handled through the google-genai SDK which provides access to the Gemini 2.5 Flash model. The user interface is built with Gradio for the monolithic deployment and Flask for the distributed architecture. Redis serves as the message queue for coordinating distributed services.

Development tools include pytest for automated testing with coverage reporting, python-dotenv for environment configuration management, and Git for version control with GitHub hosting the repository.

## Installation and Setup

Begin by cloning the repository from GitHub using git clone https://github.com/AleksandraNowinska/SoftwareEng.git and navigating to the project directory. Create a Python virtual environment using python -m venv .venv to isolate dependencies. Activate the environment with source .venv/bin/activate on macOS and Linux or .venv\Scripts\activate on Windows. Install all required packages by running pip install -r requirements.txt.

The system requires the Gemini API for generating artwork descriptions. To enable this functionality, obtain an API key from https://aistudio.google.com/apikey and create a file named .env in the project root directory. Add the line GOOGLE_API_KEY=your_api_key_here with your actual key. The application will function without this configuration by using template descriptions, but the Gemini integration provides significantly richer content with 300 to 400 word structured analyses.

## Running the Application

For monolithic deployment, execute python app.py from the project root directory after activating the virtual environment. The Gradio interface will start on localhost port 7860. Access the application by opening http://localhost:7860 in a web browser. This mode is recommended for development and demonstration purposes.

The distributed architecture requires Redis as a message broker. On macOS, install Redis using brew install redis and start the service with brew services start redis. On Linux systems, use sudo apt-get install redis-server followed by sudo systemctl start redis. Alternatively, run Redis in Docker using docker run -d -p 6379:6379 redis:alpine.

Once Redis is running, start the distributed system by executing ./distributed/start_system.sh from the project root. This script launches the orchestrator service, AI server, and interface server in sequence. Alternatively, run each component manually in separate terminal windows: python distributed/orchestrator.py, then python distributed/ai_server.py, and finally python distributed/interface_server.py. The distributed interface is accessible at http://localhost:5000.

## Testing

The test suite consists of 29 automated tests covering unit and integration scenarios. Execute all tests by running pytest tests/ -v from the project root directory. For coverage analysis, use pytest tests/ --cov=app --cov-report=html which generates an HTML coverage report in the htmlcov directory. Individual test files can be run separately using pytest tests/test_unit.py -v for unit tests or pytest tests/test_integration.py -v for integration tests.

The test coverage exceeds 85% across all modules. Unit tests verify individual components including embedding generation, image preprocessing, vector similarity search, and metadata retrieval. Integration tests validate the complete recognition pipeline from image upload through description generation. The test suite also includes validation of the Gemini API integration with appropriate fallback behavior when API keys are not configured.

## Documentation

Comprehensive project documentation is available in FINAL_REPORT.md which covers all seven development tasks including problem definition, data preparation, software requirements, agile methodology, testing strategy, distributed architecture implementation, and value sensitive design analysis. The Software Requirements Specification document in our_tasks_and_solutions/Task_3_SRS_Document.md details functional and non-functional requirements. Documentation specific to the distributed architecture is maintained in distributed/README.md. All Python modules include detailed docstrings following standard conventions for functions, classes, and modules.

## Performance Metrics

The recognition system achieves 87% accuracy for top-one predictions and 96% accuracy for top-five predictions on the test dataset. End-to-end response time from image upload to description generation averages between eight and ten seconds. The distributed architecture has been tested to support at least 50 concurrent users without performance degradation. Automated test coverage exceeds 85% across all modules including unit tests for individual components and integration tests for complete workflows.

## Environmental Impact

The system's carbon footprint is estimated at approximately 150 kilograms of CO2 per year under moderate usage of 1000 requests daily. This calculation accounts for inference operations consuming approximately 0.001 kilowatt-hours per request. The use of pre-trained CLIP models avoids training emissions which typically exceed 100 tons of CO2 for comparable vision models. Future optimizations could include model quantization to reduce computational requirements, edge deployment to minimize data transfer, and carbon-aware scheduling to run computations during periods of low-carbon electricity availability. Detailed environmental analysis is provided in FINAL_REPORT.md.

## References

The dataset is based on the Best Artworks of All Time collection available on Kaggle at https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time/data and the WikiArt Dataset at https://www.kaggle.com/datasets/steubk/wikiart. The implementation uses the OpenAI CLIP model from https://github.com/openai/CLIP for image embeddings, FAISS vector search library from https://github.com/facebookresearch/faiss, and the Gemini API from https://ai.google.dev/ for description generation. Key literature references include Building Intelligent Systems: A Guide to Machine Learning Engineering by Geert Hulten (2018) and Pro Git by Scott Chacon and Ben Straub (2020).
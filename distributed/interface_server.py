import os
import time
import csv
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import redis
from PIL import Image
import io
import base64

app = Flask(__name__)

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
LOG_PATH = "app/logs/telemetry.csv"
REQUEST_QUEUE = "artguide:requests"
RESPONSE_PREFIX = "artguide:response:"

# Initialize Redis connection (orchestrator)
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False)

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "request_id", "artist", "confidence", "response_time", "status"])


def validate_image(image_data):
    """
    Validate uploaded image data.
    
    Args:
        image_data: Binary image data
        
    Returns:
        tuple: (is_valid, error_message, PIL.Image or None)
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        
        # Check format
        if img.format not in ['JPEG', 'PNG', 'JPG']:
            return False, "Invalid format. Only JPEG and PNG are supported.", None
        
        # Check dimensions
        width, height = img.size
        if width < 50 or height < 50:
            return False, "Image too small. Minimum size is 50x50 pixels.", None
        
        if width > 5000 or height > 5000:
            return False, "Image too large. Maximum size is 5000x5000 pixels.", None
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return True, None, img
    
    except Exception as e:
        return False, f"Failed to process image: {str(e)}", None


def log_request(request_id, artist, confidence, response_time, status):
    """Log request telemetry to CSV."""
    try:
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                request_id,
                artist,
                confidence,
                response_time,
                status
            ])
    except Exception as e:
        print(f"Error logging request: {e}")


@app.route('/')
def index():
    """Render simple web interface."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Art Guide - Interface Server</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .upload-form { margin: 20px 0; padding: 20px; border: 2px dashed #ccc; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background: #0056b3; }
            #result { margin-top: 20px; padding: 20px; background: #f8f9fa; display: none; }
            .error { color: red; }
            .success { color: green; }
        </style>
    </head>
    <body>
        <h1>🖼️ Art Guide - Distributed System</h1>
        <p>Upload an artwork image to receive AI-powered recognition and description.</p>
        
        <div class="upload-form">
            <input type="file" id="imageInput" accept="image/jpeg,image/png">
            <button onclick="uploadImage()">Recognize Artwork</button>
        </div>
        
        <div id="result"></div>
        
        <script>
            async function uploadImage() {
                const input = document.getElementById('imageInput');
                const resultDiv = document.getElementById('result');
                
                if (!input.files || !input.files[0]) {
                    alert('Please select an image first');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', input.files[0]);
                
                resultDiv.innerHTML = '<p>Processing...</p>';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/api/recognize', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        resultDiv.innerHTML = `
                            <h3 class="success">Recognition Successful</h3>
                            <p><strong>Artist:</strong> ${data.artist}</p>
                            <p><strong>Title:</strong> ${data.title}</p>
                            <p><strong>Period:</strong> ${data.period}</p>
                            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                            <p><strong>Description:</strong> ${data.description}</p>
                            <p><em>Response time: ${data.response_time}s</em></p>
                        `;
                    } else {
                        resultDiv.innerHTML = `<p class="error">Error: ${data.message}</p>`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/api/recognize', methods=['POST'])
def recognize():
    """
    Handle artwork recognition requests.
    Validates input, sends to orchestrator, waits for AI server response.
    """
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    
    # Check if image is in request
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image provided'
        }), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'Empty filename'
        }), 400
    
    # Read and validate image
    image_data = file.read()
    is_valid, error_msg, img = validate_image(image_data)
    
    if not is_valid:
        log_request(request_id, "N/A", 0.0, time.time() - start_time, "validation_error")
        return jsonify({
            'status': 'error',
            'message': error_msg
        }), 400
    
    # Prepare request for AI server via orchestrator
    request_payload = {
        'request_id': request_id,
        'image': base64.b64encode(image_data).decode('utf-8'),
        'timestamp': datetime.now().isoformat(),
        'show_context': request.form.get('show_context', 'false').lower() == 'true'
    }
    
    try:
        # Send to orchestrator (Redis queue)
        redis_client.rpush(REQUEST_QUEUE, json.dumps(request_payload))
        
        # Wait for response (with timeout)
        timeout = 30  # seconds
        response_key = f"{RESPONSE_PREFIX}{request_id}"
        
        for _ in range(timeout * 10):  # Check every 100ms
            response_data = redis_client.get(response_key)
            if response_data:
                # Parse response
                response = json.loads(response_data)
                redis_client.delete(response_key)  # Clean up
                
                response_time = time.time() - start_time
                
                # Log successful request
                log_request(
                    request_id,
                    response.get('artist', 'Unknown'),
                    response.get('confidence', 0.0),
                    response_time,
                    'success'
                )
                
                return jsonify({
                    'status': 'success',
                    'artist': response.get('artist', 'Unknown'),
                    'title': response.get('title', 'Unknown'),
                    'period': response.get('period', 'Unknown'),
                    'confidence': response.get('confidence', 0.0),
                    'description': response.get('description', ''),
                    'response_time': round(response_time, 2),
                    'request_id': request_id
                })
            
            time.sleep(0.1)
        
        # Timeout
        log_request(request_id, "N/A", 0.0, time.time() - start_time, "timeout")
        return jsonify({
            'status': 'error',
            'message': 'Request timeout - AI server not responding'
        }), 504
    
    except Exception as e:
        log_request(request_id, "N/A", 0.0, time.time() - start_time, "error")
        return jsonify({
            'status': 'error',
            'message': f'Server error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    try:
        redis_client.ping()
        return jsonify({
            'status': 'healthy',
            'orchestrator': 'connected',
            'timestamp': datetime.now().isoformat()
        })
    except:
        return jsonify({
            'status': 'unhealthy',
            'orchestrator': 'disconnected',
            'timestamp': datetime.now().isoformat()
        }), 503


if __name__ == '__main__':
    print("Starting Interface Server on port 5000...")
    print(f"Orchestrator (Redis): {REDIS_HOST}:{REDIS_PORT}")
    app.run(host='0.0.0.0', port=5000, debug=False)

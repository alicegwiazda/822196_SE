import redis
import os

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))

def setup_redis():
    """
    Setup Redis as orchestrator.
    Creates necessary data structures and verifies connection.
    """
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        
        # Test connection
        client.ping()
        print(f"✓ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        
        # Clear old queues (optional - comment out in production)
        client.delete("artguide:requests")
        print("✓ Cleared request queue")
        
        # Set up queue monitoring
        info = client.info()
        print(f"✓ Redis version: {info['redis_version']}")
        print(f"✓ Used memory: {info['used_memory_human']}")
        
        print("\nOrchestrator (Redis) is ready!")
        print("Queue name: artguide:requests")
        print("Response pattern: artguide:response:<request_id>")
        
        return True
    
    except redis.ConnectionError as e:
        print(f"✗ Failed to connect to Redis: {e}")
        print(f"\nTo start Redis:")
        print("  - macOS: brew install redis && brew services start redis")
        print("  - Linux: sudo apt-get install redis-server && sudo systemctl start redis")
        print("  - Docker: docker run -d -p 6379:6379 redis:alpine")
        return False
    
    except Exception as e:
        print(f"✗ Error setting up orchestrator: {e}")
        return False


def monitor_queue():
    """Monitor the request queue in real-time."""
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        
        print("Monitoring orchestrator queue (Ctrl+C to stop)...")
        print("-" * 60)
        
        while True:
            # Get queue length
            queue_len = client.llen("artguide:requests")
            
            # Get active response keys
            response_keys = client.keys("artguide:response:*")
            
            print(f"\rQueue length: {queue_len} | Active responses: {len(response_keys)}", end="")
            
            import time
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    
    except Exception as e:
        print(f"\nError monitoring queue: {e}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'monitor':
        monitor_queue()
    else:
        setup_redis()

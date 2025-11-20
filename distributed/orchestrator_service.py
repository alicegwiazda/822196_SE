import os
import json
import time
import redis
from datetime import datetime
from collections import defaultdict
import signal
import sys

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REQUEST_QUEUE = "artguide:requests"
RESPONSE_PREFIX = "artguide:response:"
METRICS_KEY = "artguide:metrics"
ORCHESTRATOR_PORT = int(os.getenv('ORCHESTRATOR_PORT', 6380))


class OrchestratorService:
    """
    Main orchestrator service class.
    
    Responsibilities:
    - Monitor request/response flow
    - Track AI server health and performance
    - Implement load balancing (when multiple AI servers)
    - Provide metrics and monitoring API
    - Handle graceful shutdown
    """
    
    def __init__(self):
        """Initialize the orchestrator service."""
        self.redis_client = redis.Redis(
            host=REDIS_HOST, 
            port=REDIS_PORT, 
            db=0, 
            decode_responses=True
        )
        self.running = True
        self.stats = defaultdict(int)
        
        # Register shutdown handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
    
    def shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        print("\n\nShutting down orchestrator service...")
        self.running = False
        self.save_metrics()
        sys.exit(0)
    
    def verify_redis_connection(self):
        """Verify connection to Redis."""
        try:
            self.redis_client.ping()
            print(f"✓ Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
            return True
        except redis.ConnectionError as e:
            print(f"✗ Failed to connect to Redis: {e}")
            print("\nTo start Redis:")
            print("  macOS:  brew install redis && brew services start redis")
            print("  Linux:  sudo apt-get install redis-server && sudo systemctl start redis")
            print("  Docker: docker run -d -p 6379:6379 redis:alpine")
            return False
    
    def initialize_queue(self):
        """Initialize queue structure and clear stale data."""
        print("Initializing queue structures...")
        
        # Clear old response keys (cleanup)
        pattern = f"{RESPONSE_PREFIX}*"
        stale_keys = list(self.redis_client.scan_iter(match=pattern))
        if stale_keys:
            self.redis_client.delete(*stale_keys)
            print(f"  Cleaned up {len(stale_keys)} stale response keys")
        
        # Get current queue size
        queue_size = self.redis_client.llen(REQUEST_QUEUE)
        print(f"  Current queue size: {queue_size} requests")
        
        print("✓ Queue initialized")
    
    def monitor_flow(self):
        """
        Monitor request/response flow in real-time.
        
        This is the main monitoring loop that tracks:
        - Incoming requests
        - Completed responses
        - Queue backlog
        - AI server processing time
        """
        print("\n" + "=" * 70)
        print("Orchestrator Service - Active Monitoring")
        print("=" * 70)
        print("Monitoring request/response flow... (Press Ctrl+C to stop)\n")
        
        last_queue_size = 0
        last_check = time.time()
        
        while self.running:
            try:
                # Get queue metrics
                queue_size = self.redis_client.llen(REQUEST_QUEUE)
                response_keys = list(self.redis_client.scan_iter(match=f"{RESPONSE_PREFIX}*"))
                active_responses = len(response_keys)
                
                # Calculate throughput
                current_time = time.time()
                time_elapsed = current_time - last_check
                
                if queue_size != last_queue_size:
                    delta = queue_size - last_queue_size
                    direction = "↑" if delta > 0 else "↓"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Queue: {queue_size} {direction} | "
                          f"Active: {active_responses} | "
                          f"Delta: {abs(delta)}")
                    
                    # Update stats
                    if delta < 0:  # Requests processed
                        self.stats['total_processed'] += abs(delta)
                    else:  # New requests
                        self.stats['total_received'] += delta
                
                last_queue_size = queue_size
                last_check = current_time
                
                # Update metrics in Redis
                if self.stats['total_processed'] > 0:
                    metrics = {
                        'total_received': self.stats['total_received'],
                        'total_processed': self.stats['total_processed'],
                        'current_queue_size': queue_size,
                        'active_responses': active_responses,
                        'last_update': datetime.now().isoformat()
                    }
                    self.redis_client.hset(METRICS_KEY, mapping=metrics)
                
                time.sleep(1)  # Check every second
            
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def save_metrics(self):
        """Save final metrics before shutdown."""
        if self.stats['total_processed'] > 0:
            print("\n" + "=" * 70)
            print("Session Statistics:")
            print("=" * 70)
            print(f"Total requests received: {self.stats['total_received']}")
            print(f"Total requests processed: {self.stats['total_processed']}")
            print(f"Average queue size: {self.stats.get('avg_queue_size', 0)}")
            print("=" * 70)
    
    def get_metrics(self):
        """Get current metrics from Redis."""
        metrics = self.redis_client.hgetall(METRICS_KEY)
        return metrics
    
    def run(self):
        """Main execution method."""
        print("=" * 70)
        print("Art Guide - Orchestrator Service")
        print("=" * 70)
        
        # Verify Redis connection
        if not self.verify_redis_connection():
            return 1
        
        # Initialize queue structures
        self.initialize_queue()
        
        # Display configuration
        print("\nConfiguration:")
        print(f"  Request Queue: {REQUEST_QUEUE}")
        print(f"  Response Pattern: {RESPONSE_PREFIX}<request_id>")
        print(f"  Metrics Key: {METRICS_KEY}")
        print(f"  Redis: {REDIS_HOST}:{REDIS_PORT}")
        
        # Start monitoring
        self.monitor_flow()
        
        return 0


def main():
    """Entry point for orchestrator service."""
    service = OrchestratorService()
    return service.run()


if __name__ == '__main__':
    exit(main())

#!/bin/bash
echo "Starting Art Guide Distributed System"

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Redis is not running - starting redis..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # for macos
        brew services start redis 2>/dev/null || redis-server --daemonize yes
    else
        # for linux
        sudo systemctl start redis 2>/dev/null || redis-server --daemonize yes
    fi
    
    sleep 2
    
    if ! redis-cli ping > /dev/null 2>&1; then
        echo "Failed to start Redis. Please start it manually:"
        echo "macOS: brew services start redis"
        echo "Linux: sudo systemctl start redis"
        echo "Docker: docker run -d -p 6379:6379 redis:alpine"
        exit 1
    fi
fi

echo "Redis is running"

echo ""
echo "Setting up orchestrator (initial setup)"
python distributed/orchestrator.py

echo ""
echo "Starting orchestrator service"
python distributed/orchestrator_service.py &
ORCHESTRATOR_PID=$!
echo "Orchestrator service started (PID: $ORCHESTRATOR_PID)"

# Start AI Server in background
echo ""
echo "Starting AI server"
python distributed/ai_server.py &
AI_SERVER_PID=$!
echo "AI server started (PID: $AI_SERVER_PID)"

# Wait a moment for servers to initialize
sleep 3

# Start Interface Server
echo ""
echo "Starting interface server..."
echo "Interface: http://localhost:5000"
echo "AI server: Running in background (PID: $AI_SERVER_PID)"
echo "Orchestrator service: Running in background (PID: $ORCHESTRATOR_PID)"
echo "Redis queue: localhost:6379"
echo ""

python distributed/interface_server.py

# Cleanup on exit
echo ""
echo "Shutting down all services"
kill $AI_SERVER_PID 2>/dev/null
kill $ORCHESTRATOR_PID 2>/dev/null
echo "All services stopped"

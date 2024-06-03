#!/bin/bash

# Define the list of ports to kill processes for
ports=(5000 5003 5010 5013)

# Iterate over each port and attempt to kill the associated process
for port in "${ports[@]}"; do
    pid=$(lsof -t -i:$port)
    if [ -n "$pid" ]; then
        kill -9 $pid
        echo "Process with PID $pid on port $port killed successfully."
    else
        echo "No process found running on port $port."
    fi
done

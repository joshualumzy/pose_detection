#!/bin/bash

if [ ! -d "venv" ]; then
  python3 -m venv venv
  ./venv/bin/pip install -r requirements.txt
fi

echo "Running Pose Detection..."
./venv/bin/python main.py
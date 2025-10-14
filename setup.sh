#!/bin/bash

# exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting environment setup...${NC}"

# install uv if necessary
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv is not installed. Installing uv...${NC}"
    curl -LsSf https://github.com/astral-sh/uv/releases/latest/download/uv-installer.sh | sh
fi

echo -e "${YELLOW}Creating new Python environment: argue-viz${NC}"
uv venv argue-viz
source argue-viz/bin/activate

echo -e "${YELLOW}Installing dependencies...${NC}"
uv pip install -r requirements.txt
echo -e "${GREEN}Done!${NC}"
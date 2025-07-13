#!/bin/bash

# Sentiment Analysis API - cURL Examples
# Make sure the API server is running on http://localhost:8000

BASE_URL="http://localhost:8000"

echo "🎤 Sentiment Analysis API - cURL Examples"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
    echo ""
}

# Function to print command
print_command() {
    echo -e "${YELLOW}Command:${NC}"
    echo "$1"
    echo ""
}

# Function to print response
print_response() {
    echo -e "${GREEN}Response:${NC}"
    echo "$1"
    echo ""
}

# 1. Health Check
print_section "1. Health Check"
print_command "curl -X GET \"$BASE_URL/health\""
response=$(curl -s -X GET "$BASE_URL/health")
print_response "$response"

# 2. System Status
print_section "2. System Status"
print_command "curl -X GET \"$BASE_URL/status\""
response=$(curl -s -X GET "$BASE_URL/status")
print_response "$response"

# 3. Supported Languages
print_section "3. Supported Languages"
print_command "curl -X GET \"$BASE_URL/languages\""
response=$(curl -s -X GET "$BASE_URL/languages")
print_response "$response"

# 4. Text Analysis Examples
print_section "4. Text Analysis Examples"

# English text analysis
echo -e "${YELLOW}English Text Analysis:${NC}"
print_command "curl -X POST \"$BASE_URL/analyze/text\" --data-urlencode \"text=I am absolutely delighted with the service!\" --data-urlencode \"language=en\""
response=$(curl -s -X POST "$BASE_URL/analyze/text" --data-urlencode "text=I am absolutely delighted with the service!" --data-urlencode "language=en")
print_response "$response"

# Neutral text analysis
echo -e "${YELLOW}Neutral Text Analysis:${NC}"
print_command "curl -X POST \"$BASE_URL/analyze/text\" --data-urlencode \"text=The product arrived on time and was as described.\" --data-urlencode \"language=en\""
response=$(curl -s -X POST "$BASE_URL/analyze/text" --data-urlencode "text=The product arrived on time and was as described." --data-urlencode "language=en")
print_response "$response"

# Negative text analysis
echo -e "${YELLOW}Negative Text Analysis:${NC}"
print_command "curl -X POST \"$BASE_URL/analyze/text\" --data-urlencode \"text=I'm very disappointed with the quality.\" --data-urlencode \"language=en\""
response=$(curl -s -X POST "$BASE_URL/analyze/text" --data-urlencode "text=I'm very disappointed with the quality." --data-urlencode "language=en")
print_response "$response"

# 5. Audio Analysis (if sample file exists)
print_section "5. Audio Analysis"
if [ -f "incoming/Shoppee_Happy_1.opus" ]; then
    echo -e "${YELLOW}Audio File Analysis:${NC}"
    print_command "curl -X POST \"$BASE_URL/analyze\" -F \"file=@incoming/Shoppee_Happy_1.opus\" -F \"language=en\""
    response=$(curl -s -X POST "$BASE_URL/analyze" -F "file=@incoming/Shoppee_Happy_1.opus" -F "language=en")
    print_response "$response"
else
    echo -e "${RED}Sample audio file not found: incoming/Shoppee_Happy_1.opus${NC}"
    echo "Skipping audio analysis example"
fi

# 6. Get All Analyses
print_section "6. Get All Analyses"
print_command "curl -X GET \"$BASE_URL/analyses?limit=5\""
response=$(curl -s -X GET "$BASE_URL/analyses?limit=5")
print_response "$response"

# 7. Get Statistics
print_section "7. System Statistics"
print_command "curl -X GET \"$BASE_URL/statistics\""
response=$(curl -s -X GET "$BASE_URL/statistics")
print_response "$response"

# 8. Get Specific Analysis (if available)
print_section "8. Get Specific Analysis"
print_command "curl -X GET \"$BASE_URL/analyses/1\""
response=$(curl -s -X GET "$BASE_URL/analyses/1")
print_response "$response"

# 9. Delete Analysis (commented out for safety)
print_section "9. Delete Analysis (Example)"
echo -e "${YELLOW}Note: This command is commented out for safety${NC}"
print_command "# curl -X DELETE \"$BASE_URL/analyses/1\""
echo "Uncomment the above line to actually delete an analysis"

# 10. Graceful Shutdown (commented out for safety)
print_section "10. Graceful Shutdown (Example)"
echo -e "${YELLOW}Note: This command is commented out for safety${NC}"
print_command "# curl -X POST \"$BASE_URL/shutdown\""
echo "Uncomment the above line to gracefully shutdown the server"

echo ""
echo -e "${GREEN}✅ All cURL examples completed!${NC}"
echo ""
echo -e "${BLUE}Additional Notes:${NC}"
echo "- Replace 'localhost:8000' with your actual server URL if different"
echo "- For production use, consider adding authentication headers"
echo "- The graceful shutdown endpoint will terminate the server process"
echo "- Audio analysis requires valid audio files (WAV, MP3, FLAC, M4A, OPUS)" 
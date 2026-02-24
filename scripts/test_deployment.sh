#!/bin/bash
echo "Testing API..."

# 1. PTP Example
echo "\n--- 1. PTP Test ---"
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "transcript": "agent: hello sir payment is due. borrower: yes i will pay 5000 tomorrow for sure.",
           "current_date": "2024-10-25"
         }' | jq

# 2. Wrong Number Example
echo "\n--- 2. Wrong Number Test ---"
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "transcript": "agent: hello ramesh? borrower: wrong number this is suresh.",
           "current_date": "2024-10-25"
         }' | jq

# 3. Callback Later Example
echo "\n--- 3. Callback Later Test ---"
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "transcript": "agent: hello sir. borrower: i am driving right now call me in 1 hour.",
           "current_date": "2024-10-25"
         }' | jq

# 4. Real Test Data (Language Barrier / Silence)
echo "\n--- 4. Real Test Data (Hindi/Silence) ---"
curl -X POST "http://localhost:8080/predict"      -H "Content-Type: application/json"      -d '{
           "transcript": "Agent: Hello Borrower: hello Agent: Ma am आवाज़ नहीं आ रही है सिंधी बात कर रही हो ना Borrower: हैं? Agent: सिंधी बात कर रही Borrower: हो? Agent: कहां से आ रही Borrower: है? Hello, सिंधी बात कर रही है Agent: क्या? आपको callback आती है आवाज़ नहीं आ रही",
           "current_date": "2024-10-25"
         }' | jq

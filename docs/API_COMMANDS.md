# Disposition API - Quick Reference

## Server Management

### Start the API
```bash
nohup /home/ubuntu/disposition_model/venv/bin/python3 /home/ubuntu/disposition_model/app_legacy.py > app_legacy.log 2>&1 &
```

### Stop the API
```bash
ps aux | grep app_legacy.py | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

### Check if API is running
```bash
curl -s http://localhost:8005/health
```

### View logs
```bash
tail -f /home/ubuntu/disposition_model/app_legacy.log
```

---

## Test Commands (curl)

### 1. Hindi PTP - ₹2000 partial commitment
```bash
curl -s -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Agent: Aapka ₹10,000 due hai. Kab tak bharenge? Borrower: Dekho, abhi mere paas sirf ₹2000 hain. Woh main aaj sham tak bhenj deta hoon. Baaki ka balance main next month dekhunga."}' | python3 -m json.tool
```
**Expected:** `PTP`, amount: `2000`

### 2. Language Barrier (Telugu)
```bash
curl -s -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Agent: Hello, I am calling regarding your overdue EMI. When can you pay? Borrower: Naku Hindi raadu sir, Telugu lo matladandi. Evaraina unnara? Agent: Sir, please pay your amount. Borrower: Naku ardham kavadam ledu."}' | python3 -m json.tool
```
**Expected:** `LANGUAGE_BARRIER`

### 3. Settlement Request
```bash
curl -s -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Agent: Sir your outstanding is ₹45000. Borrower: I want to settle this. What is the minimum amount I can pay to close this account? Agent: Sir you need to pay at least 60 percent. Borrower: Ok I will pay ₹27000 and close it. Send me the settlement letter."}' | python3 -m json.tool
```
**Expected:** `SETTLEMENT`

### 4. Denied to Pay
```bash
curl -s -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Agent: Sir, your payment is pending since last month. Borrower: I am not going to pay this. You guys charged me extra interest which was not discussed. I already sent an email to your support. Do not call me again."}' | python3 -m json.tool
```
**Expected:** `DENIED_TO_PAY`

### 5. Wrong Number
```bash
curl -s -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Agent: Kya main Rahul se baat kar sakta hoon? Borrower: Nahi, aap galat number pe call kar rahe hain. Yeh number mera hai aur main Rahul ko nahi jaanta."}' | python3 -m json.tool
```
**Expected:** `WRONG_NUMBER`

### 6. Call Back Later
```bash
curl -s -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Agent: Hello, regarding your loan... Borrower: Abhi busy hoon, kal call karo. Agent: Sir, sirf 2 minute lagenge. Borrower: Nahi, bola na busy hoon. Kal 10 baje ke baad call karna, tab baat karenge."}' | python3 -m json.tool
```
**Expected:** `CALL_BACK_LATER`

### 7. Family Member Answered
```bash
curl -s -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Agent: Kya main Amit ji se baat kar sakta hoon? Borrower: Amit toh ghar pe nahi hai. Main unka bhai bol raha hoon. Agent: Unka payment ka message dena tha. Borrower: Woh sham ko aayenge, tab phone karna."}' | python3 -m json.tool
```
**Expected:** `ANSWERED_BY_FAMILY_MEMBER`

---

## Dashboard Links
- **Grafana:** http://65.0.97.13:3000
- **API Health:** http://65.0.97.13:8005/health
- **Prometheus:** http://65.0.97.13:9090
- **API Metrics:** http://65.0.97.13:8005/metrics

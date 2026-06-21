# Weather Blockchain Pipeline

A Weather Data Pipeline integrated with Blockchain concepts to ensure transparent, verifiable, and tamper-resistant storage of weather records.

The system automatically collects weather information from external APIs, processes the data, generates cryptographic hashes, and stores records in a blockchain-like structure for integrity verification.

---

## Features

### Current Features

* Automated weather data collection
* Data preprocessing and validation
* Blockchain-based record storage
* SHA-256 hash generation
* Block linking through previous hashes
* JSON-based data persistence
* Historical weather record tracking
* GitHub Actions automation
* Daily scheduled execution

---

## Project Structure

```text
weather-blockchain-pipeline/
│
├── blockchain/
│   ├── block.py
│   ├── blockchain.py
│
├── weather/
│   ├── weather_fetcher.py
│   ├── weather_processor.py
│
├── data/
│   ├── blockchain_data.json
│
├── tests/
│   ├── test_blockchain.py
│
├── .github/
│   └── workflows/
│       └── weather_pipeline.yml
│
├── main.py
├── requirements.txt
└── README.md
```

---

## How It Works

### Step 1: Fetch Weather Data

The application retrieves weather information from a weather API.

Example:

```json
{
  "temperature": 32,
  "humidity": 68,
  "pressure": 1008,
  "timestamp": "2026-06-21T12:00:00"
}
```

---

### Step 2: Create a Block

The weather record becomes the payload of a blockchain block.

Each block contains:

* Index
* Timestamp
* Weather data
* Previous block hash
* Current block hash

---

### Step 3: Generate Hash

SHA-256 hashing is used to create a unique fingerprint of each block.

```python
hash = sha256(block_contents)
```

---

### Step 4: Chain Verification

Every block references the hash of the previous block.

This ensures:

* Data integrity
* Tamper detection
* Immutable history

---

### Step 5: Automated Execution

GitHub Actions runs the pipeline automatically on a schedule.

Current schedule:

```yaml
cron: '30 18 * * *'
```

Runs daily at:

```text
12:00 AM IST
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/weather-blockchain-pipeline.git

cd weather-blockchain-pipeline
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python main.py
```

---

## Running Tests

```bash
pytest
```

---

## Example Blockchain Record

```json
{
  "index": 1,
  "timestamp": "2026-06-21T12:00:00",
  "weather_data": {
    "temperature": 32,
    "humidity": 68
  },
  "previous_hash": "0000000000",
  "hash": "3fd4c5b7a9..."
}
```

---

## Future Enhancements

### Weather Intelligence

* Multi-city weather monitoring
* Weather trend analysis
* Extreme weather detection
* Forecast accuracy tracking

### Blockchain Enhancements

* Proof-of-Work mechanism
* Merkle Trees
* Digital Signatures
* Smart Contract Integration

### Data Engineering

* PostgreSQL integration
* ETL pipeline improvements
* Data warehouse support
* Real-time streaming

### Cloud & DevOps

* Docker deployment
* Kubernetes orchestration
* CI/CD improvements
* Cloud deployment

### Machine Learning

* Weather prediction models
* Anomaly detection
* Climate trend forecasting
* Decision intelligence system

---

## Technologies Used

* Python
* Blockchain Fundamentals
* SHA-256 Cryptography
* REST APIs
* JSON Storage
* GitHub Actions
* Pytest

---

## Learning Objectives

This project demonstrates:

* Blockchain fundamentals
* Data pipelines
* API integration
* Data validation
* Hashing and cryptography
* Automation with GitHub Actions
* Software engineering best practices

---

## Author

Ritwik Singh

Data Science | Machine Learning | Blockchain | Software Engineering

---

## License

This project is licensed under the MIT License.

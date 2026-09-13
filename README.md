\# 🤖 Support AI



An AI-powered customer support ticket intelligence system that combines \*\*machine learning, semantic search, RAG, and a local LLM chatbot\*\* to analyze and assist with customer support tickets.



The system can automatically classify incoming tickets, predict their priority and support queue, retrieve similar historical tickets, and provide troubleshooting guidance through an interactive chatbot.



\---



\## ✨ Features



\### 🎯 AI Ticket Triage



The system analyzes a support ticket and predicts:



\* \*\*Ticket Category\*\*

\* \*\*Support Queue\*\*

\* \*\*Priority\*\*

\* \*\*Queue Confidence\*\*



The triage system uses trained machine learning models built with \*\*TF-IDF and Logistic Regression\*\*.



\---



\### 🔎 Semantic Ticket Search



Support AI uses \*\*Retrieval-Augmented Generation (RAG)\*\* to find historically similar support tickets.



The system:



1\. Converts support tickets into vector embeddings.

2\. Stores the embeddings in a \*\*FAISS\*\* index.

3\. Converts a user's support question into an embedding.

4\. Searches for the most semantically similar tickets.

5\. Uses the retrieved tickets to provide relevant troubleshooting guidance.



This allows the system to find tickets based on \*\*meaning\*\*, rather than relying only on exact keyword matches.



\---



\### 💬 AI Support Chatbot



The chatbot uses a locally running \*\*Qwen3 4B\*\* model through \*\*Ollama\*\*.



The LLM acts as a routing component that determines whether a user request requires:



\* `SEARCH\_TICKETS`

\* `QUERY\_STATS`



For customer support problems, the system searches the historical ticket database.



For dataset-related questions, the system can return statistics such as ticket counts by support queue.



Non-support questions are rejected instead of being answered by the system.



\---



\### 📊 Ticket Statistics



The chatbot can answer questions about the support ticket dataset, including:



\* Number of tickets

\* Number of tickets in each support queue

\* Ticket counts by priority

\* Dataset statistics



\---



\### 🖥️ Interactive Web Dashboard



The project includes a React frontend that provides:



\* AI ticket triage

\* Support chatbot

\* Classification results

\* Priority prediction

\* Assigned support team

\* Confidence score

\* Retrieved ticket sources



\---



\## 🏗️ System Architecture



```text

&#x20;                    ┌─────────────────────┐

&#x20;                    │    React Frontend   │

&#x20;                    │      Dashboard      │

&#x20;                    └──────────┬──────────┘

&#x20;                               │

&#x20;                               ▼

&#x20;                    ┌─────────────────────┐

&#x20;                    │    FastAPI Backend  │

&#x20;                    └──────────┬──────────┘

&#x20;                               │

&#x20;             ┌─────────────────┼─────────────────┐

&#x20;             │                 │                 │

&#x20;             ▼                 ▼                 ▼

&#x20;      ┌────────────┐    ┌─────────────┐   ┌─────────────┐

&#x20;      │ ML Models  │    │ RAG / FAISS │   │   Ollama    │

&#x20;      │            │    │             │   │   Qwen3 4B  │

&#x20;      └────────────┘    └─────────────┘   └─────────────┘

&#x20;             │                 │                 │

&#x20;             ▼                 ▼                 ▼

&#x20;      Category / Queue   Similar Tickets    Tool Routing

&#x20;      / Priority                             

```



\---



\## 🧠 Machine Learning



The project contains separate models for ticket classification.



\### Category Classification



A \*\*TF-IDF + Logistic Regression\*\* model is used to predict the ticket category.



Performance:



| Metric   |  Score |

| -------- | -----: |

| Accuracy | 80.66% |

| Macro F1 | 78.75% |



The dataset's `type` field is used as the ticket category.



\---



\### Queue Classification



A TF-IDF + Logistic Regression model predicts the appropriate support queue.



The system can classify tickets into queues such as:



\* Technical Support

\* Product Support

\* Customer Service

\* IT Support

\* Billing and Payments

\* Returns and Exchanges

\* Service Outages and Maintenance

\* Sales and Pre-Sales

\* Human Resources

\* General Inquiry



\---



\### Priority Classification



A separate machine learning model predicts the ticket priority.



Possible priority levels include:



\* Low

\* Medium

\* High



\---



\## 🔍 RAG Pipeline



The Retrieval-Augmented Generation component uses:



\* \*\*Sentence Transformers\*\*

\* `all-MiniLM-L6-v2`

\* \*\*FAISS\*\*

\* Cosine similarity through normalized embeddings



The ticket text is constructed from:



```text

Subject + Body + Answer

```



The resulting embeddings are normalized and stored in a FAISS `IndexFlatIP` index.



Because the embeddings are normalized, inner-product similarity corresponds to cosine similarity.



A similarity threshold is used to avoid returning unrelated tickets.



\### Example



User:



> My payment was declined. What should I do?



The system retrieves similar historical tickets and extracts relevant troubleshooting information.



Example response:



```text

Based on similar support tickets, you can try:



\- Verify the expiry date and available balance on your credit card.



Sources:

\- ticket #9052

\- ticket #12655

\- ticket #13254

```



\---



\## 💬 Chatbot Tool Routing



The chatbot uses Qwen3 to determine which backend operation should be executed.



\### SEARCH\_TICKETS



Used for customer support problems such as:



\* Login problems

\* Password problems

\* Payment problems

\* Billing issues

\* Refunds

\* Returns

\* Exchanges

\* Wrong items

\* Damaged products

\* Orders

\* Delivery issues

\* Technical problems

\* Product problems

\* Troubleshooting



\### QUERY\_STATS



Used for numerical or statistical questions about the dataset.



Example:



> How many tickets are there in each support queue?



The backend calculates the statistics directly from the database.



\---



\## 🗃️ Dataset



The project uses a multilingual customer support ticket dataset containing fields including:



```text

id

subject

body

answer

type

queue

priority

language

tag\_1

tag\_2

tag\_3

tag\_4

tag\_5

tag\_6

tag\_7

tag\_8

```



The ticket data is stored locally and used for:



\* Machine learning

\* Semantic search

\* RAG

\* Dataset statistics



\---



\## 🛠️ Technologies



\### Backend



\* Python

\* FastAPI

\* Uvicorn

\* SQLite

\* scikit-learn

\* pandas

\* NumPy

\* Joblib



\### Machine Learning



\* TF-IDF

\* Logistic Regression

\* Sentence Transformers

\* FAISS



\### AI / LLM



\* Ollama

\* Qwen3 4B

\* RAG



\### Frontend



\* React

\* Vite

\* JavaScript

\* Lucide React



\---



\## 📁 Project Structure



\### Backend



```text

final project/

│

├── api.py

│

├── final project.ipynb

│

├── dataset-tickets-multi-lang-4-20k.csv

│

├── tickets.db

│

├── tickets\_rag.index

├── rag\_metadata.pkl

│

├── clf\_category.joblib

├── clf\_queue.joblib

├── clf\_prio.joblib

│

├── tfidf\_vectorizer.joblib

├── model\_metadata.json

│

└── \_\_pycache\_\_/

```



\### Frontend



```text

frontend/

│

├── src/

│   ├── App.jsx

│   └── ...

│

├── public/

├── package.json

├── package-lock.json

├── vite.config.js

└── index.html

```



\---



\# 🚀 Installation \& Setup



\## 1. Clone the Repository



```bash

git clone <https://github.com/zeena-z/customer-support-rag-agent>

cd "final project"

```



\---



\## 2. Install Python Dependencies



Make sure Python is installed.



Then install the required packages:



```bash

pip install fastapi uvicorn pandas numpy scikit-learn joblib requests faiss-cpu sentence-transformers

```



\---



\## 3. Install Ollama



Install Ollama and download the Qwen3 model.



Then run:



```bash

ollama run qwen3:4b

```



The backend expects Ollama to be available at:



```text

http://localhost:11434

```



\---



\## 4. Start the Backend



Open a terminal and navigate to the backend directory:



```powershell

cd "C:\\Users\\zeena\\OneDrive\\Desktop\\final project"

```



Start FastAPI:



```powershell

uvicorn api:app --reload --port 8000

```



The API will run at:



```text

http://127.0.0.1:8000

```



FastAPI documentation is available at:



```text

http://127.0.0.1:8000/docs

```



\---



\## 5. Start the Frontend



Open another terminal:



```powershell

cd "C:\\Users\\zeena\\frontend"

```



Install the frontend dependencies:



```bash

npm install

```



Start the development server:



```bash

npm run dev

```



The frontend will normally be available at:



```text

http://localhost:5175/

```



\---



\## 6. Run the Complete System



The project requires three running components:



\### Terminal 1 — Ollama



```bash

ollama run qwen3:4b

```



\### Terminal 2 — FastAPI



```bash

uvicorn api:app --reload --port 8000

```



\### Terminal 3 — React



```bash

npm run dev

```



Then open the frontend in your browser.



\---



\# 🔌 API Endpoints



\## Health Check



```http

GET /health

```



Example response:



```json

{

&#x20; "status": "ok"

}

```



\---



\## Ticket Triage



```http

POST /triage

```



The endpoint accepts a support ticket and returns:



```json

{

&#x20; "category": "...",

&#x20; "predicted\_category": "...",

&#x20; "predicted\_queue": "...",

&#x20; "queue\_confidence": 0.0,

&#x20; "predicted\_priority": "..."

}

```



\---



\## Chat



```http

POST /chat

```



The chatbot accepts a conversation and returns an AI-generated support response along with the IDs of retrieved tickets.



Example:



```json

{

&#x20; "answer": "...",

&#x20; "cited\_ticket\_ids": "9052, 12655, 13254",

&#x20; "tool\_used": "SEARCH\_TICKETS"

}

```



\---



\## Dataset Statistics



```http

GET /stats

```



Returns statistics calculated from the support ticket database.



\---



\# 🧪 Example Queries



\### Support Troubleshooting



```text

I cannot log into my account.

```



The system searches similar support tickets and provides troubleshooting suggestions.



\---



\### Payment Problem



```text

My payment was declined. What should I do?

```



The system retrieves similar billing and payment tickets.



\---



\### Returns



```text

I received the wrong item in my order. I want to return it.

```



The system searches historical return and exchange tickets.



\---



\### Dataset Statistics



```text

How many tickets are there in each support queue?

```



The system returns ticket counts for each support queue.



\---



\### Unsupported Question



```text

What is the capital of France?

```



The system rejects the request because it is outside the scope of the support ticket system.



\---



\# 🔐 Notes



This project is designed as a local AI application.



The chatbot uses a locally hosted Qwen3 model through Ollama rather than relying on a cloud LLM API.



The project also uses locally stored machine learning models, a SQLite database, and a local FAISS vector index.



\---



\# 📌 Project Goals



The main goals of Support AI are to:



\* Automate initial support ticket classification.

\* Predict the appropriate support queue.

\* Identify ticket priority.

\* Reduce the time required to find similar historical tickets.

\* Provide useful troubleshooting suggestions.

\* Allow support teams to query ticket statistics.

\* Demonstrate the integration of machine learning, RAG, LLMs, APIs, and a web interface in one application.



\---



\# 👩‍💻 Project



\*\*Support AI — Customer Support Ticket Intelligence System\*\*



Built as an Artificial Intelligence \& Data Science project combining:



\*\*Machine Learning + RAG + LLM + FastAPI + React\*\*




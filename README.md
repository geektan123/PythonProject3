# API Documentation

## Overview
This API allows users to upload documents (PDF, JSON) and perform queries on them. It processes the uploaded files, stores embeddings, and provides relevant responses based on queries.

## Base URL
```
https://pythonproject-07i0.onrender.com/docs
```

---

## Endpoints

### 1. Upload Document
#### **POST** `/upload-document/`

Uploads a document (PDF or JSON) to the server for processing.

#### **Request**
**Headers:**
```
accept: application/json
Content-Type: multipart/form-data
```

**Form Data:**
```
file=@yourfile.pdf;type=application/pdf
file=@yourfile.json;type=application/json
```

#### **cURL Example:**
```sh
curl -X 'POST' \
  'https://pythonproject-07i0.onrender.com/upload-document/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@budget_speech.pdf;type=application/pdf'
```

#### **Response:**
```json
{
  "message": "Document processed and embeddings stored successfully",
  "chunk_count": 134
}
```

---

### 2. Query Document
#### **POST** `/query-document/`

Queries an uploaded document for relevant information.

#### **Request**
**Headers:**
```
accept: application/json
Content-Type: application/x-www-form-urlencoded
```

**Body Parameters:**
```
query=your_query_text
```

#### **cURL Example:**
```sh
curl -X 'POST' \
  'https://pythonproject-07i0.onrender.com/query-document/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'query=what%20is%20the%20new%20tax%20slab'
```

#### **Response:**
```json
{
  "query": "what is the new tax slab",
  "most_similar_chunk": "...",
  "similarity_score": 0.44,
  "answer": "The new tax slab announced in the budget is..."
}
```

---

### 3. Query JSON Document
#### **POST** `/query-json/`

Queries an uploaded JSON file to perform operations on numerical fields.

#### **Request**
**Headers:**
```
accept: application/json
Content-Type: application/x-www-form-urlencoded
```

**Body Parameters:**
```
operation=max
field=age
```

#### **cURL Example:**
```sh
curl -X 'POST' \
  'https://pythonproject-07i0.onrender.com/query-json/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'operation=max&field=age'
```

#### **Response:**
```json
{
  "operation": "max",
  "field": "age",
  "field_type": "numerical",
  "result": 64,
  "value_count": 50
}
```

---

## Notes
- Supported file types: **PDF, JSON**
- Ensure the JSON file contains numerical fields for `query-json` operations.
- The API supports chunking for efficient document querying.

---

## Contact
For support, please contact the API administrator.


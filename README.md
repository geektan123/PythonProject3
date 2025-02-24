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
  "most_similar_chunk": "always believed in the admirable energy and ability of the middle class in \nnation building. In recognition of their contribution, we have periodically \nreduced their tax burden. Right after 2014, the ‘Nil tax ’ slab was raised to  \n` 2.5 lakh, which was further raised to ` 5 lakh in 2019 and to ` 7 lakh in 2023. \nThis is reflective of our Government’s trust on the middle-class tax payers. I am \nnow happy to announce that there will be no income tax payable upto income \nof ` 12 lakh (i.e. average income of ` 1 lakh per month other than special rate \nincome such as capital gains) under the new regime. This limit will be ` 12.75 \nlakh for salaried tax payers, due to standard deduction of ` 75,000.  \n157. Slabs and rates are being changed across the board to benefit all tax -\npayers. The new structure will substantially reduce the taxes of the middle \nclass and leave more money in their hands, boosting household consumption, \nsavings and investment.",
  "similarity_score": 0.44562870264053345,
  "answer": "The new tax slab announced in the budget is as follows:\n\n- No income tax payable up to an income of Rs. 12 lakhs (i.e., an average income of Rs. 1 lakh per month) under the new regime.\n- For salaried taxpayers, the limit is Rs. 12.75 lakhs, owing to a standard deduction of Rs. 75,000."
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


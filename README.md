# Information-Retrieval-Project

## Wikipedia Search Engine

Hello everyone!

As part of our **Information Retrieval** course, we were asked to implement a working search engine over the **entire English Wikipedia corpus**, focusing on the **minimum required functionality**.

---

## Engine Structure

* **search_frontend.py** – Flask application that runs the search engine and handles HTTP requests.
* **backend.py** – Core backend class responsible for loading indexes and handling the retrieval and ranking process.
* **inverted_index_gcp.py** – Utilities for reading and iterating over inverted indexes stored in a GCP bucket.

---

## Index Structures

The engine uses inverted indexes built from Wikipedia articles.
Indexes are created offline using the provided GCP notebooks and stored in a Google Storage bucket.
At runtime, the backend loads these indexes and accesses the posting lists as needed.

---

## How to Use

1. Create the required indexes and update the paths in `backend.py`.
2. Run `search_frontend.py` to start the Flask server.
3. Send queries via browser or HTTP request, for example:

   ```
   http://localhost:8080/search?query=example+query
   ```

The engine can also be run locally using Google Colab, connected to the GCP bucket.

---
This project was made by Ram Shiri and Orian Rashti



ה


# Retrieval Systems with Adaptive Okapi BM25 and Boolean Search

This project implements an advanced information retrieval system using a vector space model and advanced Boolean search techniques. The system is capable of processing documents, indexing them, and efficiently retrieving relevant documents based on user queries.

## Project Details

**Author:** AmirAli Amini  
**Student ID:** 610399102  
**Project:** HW4

## Overview

In this project, we developed an information retrieval system with the following features:

1. **Document Processing**:
   - Tokenization using the `nltk` library to break down text into meaningful components.
   - Removal of stop words and punctuation for efficient indexing.

2. **Indexing**:
   - Implementation of a posting list to store document indexes and term frequencies.
   - Calculation of term frequency-inverse document frequency (TF-IDF) for each document and term.

3. **Query Handling**:
   - Advanced Boolean search capability for precise document retrieval.
   - Support for Okapi BM25 ranking with automatic mode selection based on query length.
   - Implementation of language modeling techniques using Jelinek-Mercer smoothing.

4. **Efficiency Improvements**:
   - Utilization of heap data structures to quickly retrieve top-ranked documents.
   - Automated switching between basic and advanced Okapi methods based on query characteristics.

## Key Features

### Okapi BM25 Ranking

The system supports Okapi BM25 ranking, with an automatic mode selection based on query length. This feature allows the system to adaptively choose between basic and long-form modes for efficient retrieval.

### Jelinek-Mercer Smoothing

For language modeling, Jelinek-Mercer smoothing is used, which balances between document-specific probabilities and overall language model probabilities. This approach simplifies the implementation while maintaining retrieval effectiveness.

## Parameters and Decisions

- **Lambda**: Set to 0.9 to maintain the relevance of documents to the query, ensuring the system prioritizes relevant results.
- **k1, k3, b**: Parameters for Okapi BM25 were selected based on literature and known effective values to ensure robust performance.
- **Smoothing Technique**: Jelinek-Mercer smoothing was chosen for its simplicity and effective results.

## Libraries Used

- `nltk`: For text tokenization and stop word removal.
- `numpy`: For efficient numerical operations.
- `heapq`: For managing top-k document retrieval efficiently.

## Results and Analysis

Precision-recall charts are available at the end of the project, providing insights into the system's performance. Comments within the codebase explain the functions, input parameters, outputs, and overall operation.

## Getting Started

To use the system, ensure you have the necessary libraries installed:

```bash
pip install nltk numpy
```

Then, run the `searchEngine` class to index documents and execute queries.

## Conclusion

This project demonstrates the application of advanced information retrieval techniques, showcasing a robust system for efficient and accurate document retrieval.

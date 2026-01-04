# PNU InfoSec Challenge - Final Project

This repository contains the complete source code and documentation for the PNU Information Security Challenge. The project is divided into two main tasks, covering computer vision and large language models (RAG).

## Project Structure

```
├── task1_image_retrieval/     # Task 1: Vehicle Detection and Classification
├── task2_llm_rag/             # Task 2: LLM-based RAG System for Security
└── report/                    # LaTeX Source for Final Report
```

## Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for extremely fast Python package management. Please ensure you have it installed before proceeding.

## Task 1: Image Retrieval & Analysis

This module implements a complete pipeline for analyzing vehicle traffic from video feeds, specifically tailored for the Indonesian context.

### Demo Output

[![Task 1 Demo](https://img.youtube.com/vi/IU14XfCYmOo/0.jpg)](https://youtu.be/IU14XfCYmOo)

### Pipeline Overview

1.  **Data Collection**:
    *   `scripts/download_youtube.py`: Automatically downloads CCTV footage from YouTube.
    *   `scripts/run_scraper.py`: Scrapes images of specific vehicle types (Pickup, Truck, Bus, SUV, MPV) for classification training.
2.  **Dataset Preparation**:
    *   `scripts/video_to_yolo.py`: Uses GroundingDINO to auto-annotate video frames for object detection (Teacher-Student approach).
    *   `scripts/run_cropping.py`: Uses GroundingDINO to crop vehicles from scraped images for the classification gallery.
3.  **Model Training**:
    *   `scripts/train_detection.py`: Trains YOLOv10/RT-DETR models for vehicle detection.
    *   `scripts/train_arcface.py`: Trains an ArcFace-based EfficientNet-B3 model for fine-grained vehicle classification.
4.  **Inference**:
    *   `scripts/final_experiment.py`: Integrates detection and classification to process video streams.

### Usage

To run the final demonstration pipeline:
```bash
cd task1_image_retrieval
uv sync
uv run scripts/final_experiment.py --video <path_to_video>
```

## Task 2: LLM & RAG System

This module implements a Retrieval-Augmented Generation (RAG) system designed to answer security-related queries, specifically focusing on Common Vulnerabilities and Exposures (CVEs) and PII protection. It exposes an API built with **FastAPI**.

### System Architecture

*   **Guardrails Layer**: Filters unsafe inputs and outputs (PII protection).
*   **Core Agent**: Routes queries to either the RAG tool or the LLM's internal knowledge.
*   **Vector Database**: ChromaDB stores document embeddings (using `qwen3-embedding:0.6b`).
*   **LLM**: Uses `qwen3:8b` via [Ollama](https://ollama.com/) for generation by default.
    *   *Note: If you wish to use a different provider (e.g., OpenAI, Anthropic), you can modify the initialization in `task2_llm_rag/llm.py`.*

### API Endpoints

The system provides the following endpoints:
*   `POST /chat`: Main endpoint for interacting with the LLM Agent (RAG + Guardrails).
*   `POST /rag-query`: Direct testing endpoint for retrieval-only queries.

### Usage

1.  Ensure Ollama is running with `qwen3` and `qwen3-embedding` models pulled.
2.  Start the FastAPI server:
    ```bash
    cd task2_llm_rag
    uv sync
    uv run fastapi dev main.py
    ```
3.  The API will be available at `http://localhost:8000`.

## Report

The `report/` directory contains the LaTeX source code for the final project report. It documents the methodology, experiments, and results for all tasks.

### Compilation
To compile the report:
```bash
cd report
pdflatex main.tex
```

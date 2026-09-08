# AI-Based Vision Question Answering System

An AI-based Vision Question Answering (VQA) system developed using Python and computer vision techniques. The application allows users to upload an image and ask questions about the visual content. It uses vision-language models such as CLIP and BLIP to generate and evaluate answers.

## Project Overview

Vision Question Answering combines computer vision and natural language processing to understand images and answer questions related to them.

This project provides a CPU-friendly VQA demonstration using CLIP and BLIP models. It is designed to run on systems without a dedicated GPU and provides an interactive interface using Streamlit.

## Features

- Upload an image through the web interface
- Ask natural-language questions about the uploaded image
- Generate candidate answers based on the question type
- Rank candidate answers using CLIP
- Run CLIP inference using CPU
- Use ONNX Runtime for optimized CLIP inference
- Generate open-ended answers using BLIP
- Display multiple possible BLIP answers
- Display the best answer with a confidence score in CLIP mode
- Manually modify candidate answers
- Simple and interactive Streamlit interface
- Designed for CPU-only environments

## Technologies Used

- **Python** – Main programming language
- **Streamlit** – Web-based user interface
- **PyTorch** – Deep learning framework
- **Hugging Face Transformers** – Vision-language model implementation
- **CLIP** – Image and text matching
- **BLIP** – Image captioning and visual question answering
- **ONNX Runtime** – CPU-optimized model inference
- **Optimum** – Model optimization and ONNX integration
- **Pillow** – Image processing

## System Workflow

The basic workflow of the application is:

1. User uploads an image.
2. User enters a question related to the image.
3. The uploaded image is preprocessed.
4. The question is analyzed to determine suitable answer candidates.
5. CLIP compares the image with candidate answers.
6. The answers are ranked according to their similarity scores.
7. Alternatively, BLIP generates open-ended answer suggestions.
8. The application displays the generated or ranked answers.
9. In CLIP mode, the top answer and confidence score are displayed.

## Project Structure

```text
AI-Vision-QA/
│
├── app.py
│
├── model/
│   ├── __init__.py
│   ├── blip_model.py
│   └── clip_model.py
│
├── utils/
│   ├── __init__.py
│   ├── inference.py
│   └── preprocess.py
│
├── requirements.txt
│
└── README.md

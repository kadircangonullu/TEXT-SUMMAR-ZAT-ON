# Document and Dataset Summarization App

A Streamlit-based text summarization application that can summarize uploaded PDF documents, custom text, and dataset articles. The project uses a T5-based language model and evaluates generated summaries with ROUGE metrics.

## Project Overview

The application is designed to make long-form text easier to review by generating concise summaries. It supports:

- PDF document summarization
- Custom text summarization
- CSV dataset testing
- Comparison of generated summaries with reference summaries
- ROUGE-1, ROUGE-2, and ROUGE-L evaluation

## Model

The project uses **LaMini-Flan-T5-248M** for text summarization.

Main model components:

- `T5Tokenizer` - converts input text into model-compatible tokens.
- `T5ForConditionalGeneration` - generates summaries.
- Hugging Face `pipeline` - provides the summarization workflow.

Configured summarization parameters:

- `max_length = 500`
- `min_length = 50`
- `torch_dtype = torch.float32`

## Technologies and Libraries

| Library | Purpose |
| --- | --- |
| Streamlit | Web application interface, tabs, file uploads, and visualization |
| Transformers | Loading the T5 model/tokenizer and running the summarization pipeline |
| PyPDF2 | Extracting text from PDF files page by page |
| Pandas | Loading and processing CSV datasets |
| rouge-score | Calculating ROUGE evaluation metrics |
| PyTorch | Model execution and CPU/GPU processing |
| Base64 | Encoding PDF content for HTML embedding |

## Features

### 1. PDF Summarization

Users can upload a PDF file. The application extracts its text page by page with PyPDF2 and generates a summary.

### 2. Custom Text Summarization

Users can manually enter text and generate a summary directly from the interface.

### 3. Dataset Testing

The application accepts a CSV dataset containing the following columns:

```text
article
highlights
```

For each selected row:

1. The `article` text is summarized.
2. The `highlights` value is treated as the reference summary.
3. The generated and reference summaries are compared.
4. ROUGE scores are calculated and displayed.

### 4. ROUGE Evaluation

The project evaluates summarization quality using:

- **ROUGE-1**
- **ROUGE-2**
- **ROUGE-L**

The scorer uses stemming when comparing the reference and generated summaries.

## User Interface

The Streamlit interface contains two main tabs:

- **Summarize PDF/Custom Text** - summarizes uploaded PDFs or manually entered text.
- **Test Dataset** - loads a dataset, generates summaries, and displays ROUGE scores.

## Core Functions

### Text Summarization

```python
def summarize_text(input_text):
    result = pipe_sum(input_text)
    return result[0]['summary_text']
```

### ROUGE Score Calculation

```python
def calculate_rouge(true_summary, predicted_summary):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )
    scores = scorer.score(true_summary, predicted_summary)
    return scores
```

### Dataset Validation

```python
data = pd.read_csv(dataset_path)

if 'article' not in data.columns or 'highlights' not in data.columns:
    st.error("Dataset must contain 'article' and 'highlights' columns.")
else:
    # Process dataset
    pass
```

## Installation

The report identifies the libraries used by the project, but it does not provide an exact environment file or package versions. A typical installation command for the listed dependencies is:

```bash
pip install streamlit transformers PyPDF2 pandas rouge-score torch
```

> Note: Exact dependency versions are not specified in the project report.

## Running the Application

The report describes the project as a Streamlit application but does not specify the source-code filename. If the main application file is named `app.py`, it can typically be started with:

```bash
streamlit run app.py
```

Replace `app.py` with the actual entry-point filename if it is different.

## Example Dataset Format

```csv
article,highlights
"Long article text...","Reference summary..."
```

Both `article` and `highlights` columns are required.

## Use Cases

The application can be used for:

- **Education** - summarizing lecture notes for quick review.
- **Research** - generating concise summaries of academic papers.
- **Business** - producing short summaries of reports and documents.
- **Media** - generating informative summaries of news articles.

## Project Structure

The report does not specify the repository's exact folder/file structure. Based on the described architecture, the application consists conceptually of:

```text
Input
├── PDF document
├── Custom text
└── CSV dataset
        |
        v
Text extraction / dataset processing
        |
        v
LaMini-Flan-T5-248M summarization
        |
        +--> Generated summary
        |
        └--> ROUGE evaluation (when reference summary is available)
```

## Notes

- PDF text extraction is performed page by page.
- Dataset evaluation requires `article` and `highlights` columns.
- ROUGE evaluation compares model-generated summaries against reference summaries.
- The model operates with 32-bit floating-point processing according to the report.

## Authors

- Kadir Can Gönüllü
- Derya Lara Kesti
- Emre Yeloğlu

## Academic Context

Developed as an **AI Project - Text Summarization** project for the Computer Engineering program, Faculty of Computer and Information Sciences, Adana Alparslan Türkeş Science and Technology University.

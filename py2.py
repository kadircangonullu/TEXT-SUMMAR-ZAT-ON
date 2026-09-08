import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import pandas as pd
import torch
import base64
import PyPDF2
from rouge_score import rouge_scorer

# MODEL AND TOKENIZER SETUP
# Define the language model and tokenizer to be used.
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, device_map='auto', torch_dtype=torch.float32
)

# Summarization pipeline
pipe_sum = pipeline(
    'summarization',
    model=base_model,
    tokenizer=tokenizer,
    max_length=500,  # Maximum length of the summary
    min_length=50    # Minimum length of the summary
)

# Function to summarize text
def summarize_text(input_text):
    result = pipe_sum(input_text)
    return result[0]['summary_text']

# ROUGE score calculation
def calculate_rouge(true_summary, predicted_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(true_summary, predicted_summary)
    return scores

# Streamlit interface settings
st.set_page_config(layout='wide', page_title="Summarization App with Dataset")

# Main application function
def main():
    st.title('Document and Dataset Summarization App')

    # Tabs for different functionalities
    tab1, tab2 = st.tabs(["Summarize PDF/Custom Text", "Test Dataset"])

    # Tab 1: Summarize PDF or custom text
    with tab1:
        # File uploader for PDF
        uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])

        # Text input area for custom text
        input_text = st.text_area("Or enter your text below:", height=200)

        pdf_text = ""
        if uploaded_file is not None:
            try:
                # Read the PDF file
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()

                st.success("PDF uploaded successfully. Text extracted:")
                st.text_area("Extracted PDF Text:", pdf_text, height=200)

            except Exception as e:
                st.error(f"Error reading PDF file: {e}")

        if st.button("Summarize PDF Text"):
            if pdf_text.strip():
                summary = summarize_text(pdf_text)
                st.success("PDF Summary:")
                st.write(summary)
            else:
                st.warning("No text found in the uploaded PDF.")

        if st.button("Summarize Custom Text"):
            if input_text.strip():
                summary = summarize_text(input_text)
                st.success("Summary:")
                st.write(summary)
            else:
                st.warning("Please enter some text to summarize.")

    # Tab 2: Test the dataset
    with tab2:
        st.header("Dataset Summarization and Evaluation")

        # Dataset file input
        dataset_path = st.text_input("Enter the path to your dataset (CSV):")

        if dataset_path.strip():
            try:
                # Load the dataset
                data = pd.read_csv(dataset_path)

                if 'article' not in data.columns or 'highlights' not in data.columns:
                    st.error("Dataset must contain 'article' and 'highlights' columns.")
                else:
                    # Display the first few rows of the dataset
                    st.write("First few rows of the dataset:")
                    st.dataframe(data.head())

                    # Dropdown to select a row
                    row_index = st.selectbox("Select a row to summarize", range(len(data)))

                    # Get the article and true summary for the selected row
                    selected_article = data['article'][row_index]
                    true_summary = data['highlights'][row_index]

                    # Summarize the selected article
                    predicted_summary = summarize_text(selected_article)

                    # Display the details for the selected row
                    st.write("Original Article:")
                    st.text(selected_article)

                    st.write("True Summary:")
                    st.text(true_summary)

                    st.write("Predicted Summary:")
                    st.success(predicted_summary)

                    # Calculate and display ROUGE score
                    rouge_scores = calculate_rouge(true_summary, predicted_summary)
                    st.write("ROUGE Scores:")
                    st.json(rouge_scores)

            except Exception as e:
                st.error(f"Error loading or processing dataset: {e}")
        else:
            st.warning("Please provide the path to the dataset.")



# Entry point to run the application.
if __name__ == '__main__':
    main()

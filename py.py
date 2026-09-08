import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# MODEL AND TOKENIZER SETUP
# Define the language model and tokenizer to be used.
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(
    checkpoint, device_map='auto', torch_dtype=torch.float32
)

# PDF file loading and preprocessing function
def file_preprocessing(file):
    # Load the PDF file and split it into pages using PyPDFLoader.
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    
    # Split long texts into manageable chunks using RecursiveCharacterTextSplitter.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    
    # Combine the text chunks into a single string.
    final_texts = ""
    for text in texts:
        print(text)  # Print each text chunk to the console.
        final_texts += text.page_content
    return final_texts

# Function to create a summarization pipeline with the language model
def llm_pipeline(filepath):
    # Define the summarization pipeline.
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,  # Maximum length of the summary
        min_length=50    # Minimum length of the summary
    )
    # Prepare the input text by preprocessing the file.
    input_text = file_preprocessing(filepath)
    # Generate a summary for the input text.
    result = pipe_sum(input_text)
    result = result[0]['summary_text']  # Extract the summary text.
    return result

@st.cache_data
# Function to display the PDF in the Streamlit interface
def displayPDF(file):
    # Open the file in binary mode and encode it in base64 format.
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embed the base64-encoded PDF as HTML.
    pdf_display = (
        f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
        f'width="100%" height="600" type="application/pdf"></iframe>'
    )

    # Render the embedded PDF in the Streamlit interface.
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit interface settings
st.set_page_config(layout='wide', page_title="Summarization App")

# Main application function
def main():
    st.title('Document Summarization App using Language Model')

    # Add a file uploader widget for uploading PDF files.
    uploaded_file = st.file_uploader("Upload your PDF File", type=['pdf'])

    # If a file is uploaded and the "Summarize" button is clicked, execute the summarization process.
    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)  # Create a two-column layout.
            
            # Save the uploaded file as a temporary file.
            filepath = "data/" + uploaded_file.name
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())
            
            # Display the uploaded PDF in the left column.
            with col1:
                st.info("Uploaded PDF File")
                pdf_viewer = displayPDF(filepath)

            # Display the summarization result in the right column.
            with col2:
                st.info("Summarization is below")
                summary = llm_pipeline(filepath)  # Generate the summary.
                st.success(summary)  # Display the summary.

# Entry point to run the application.
if __name__ == '__main__':
    main()

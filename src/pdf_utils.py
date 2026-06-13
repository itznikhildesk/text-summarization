import fitz  # PyMuPDF
from typing import Optional

def extract_text_from_pdf(pdf_file) -> Optional[str]:
    """
    Extracts text from a PDF file using PyMuPDF.
    
    Args:
        pdf_file: The uploaded PDF file object.
        
    Returns:
        The extracted text as a string, or None if extraction fails.
    """
    try:
        # Read the PDF file
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        
        doc.close()
        
        if not text.strip():
            return None
            
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

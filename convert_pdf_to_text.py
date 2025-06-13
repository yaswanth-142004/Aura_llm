import pdfplumber
import os

def pdf_to_txt(pdf_path, txt_output_path):
    """
    Convert a PDF file to plain text format.

    Args:
        pdf_path (str): Path to the input PDF file.
        txt_output_path (str): Path to save the extracted TXT file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    all_text = []

    # Open and extract text from each page
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Optional: clean up excessive whitespace
                clean_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
                all_text.append(clean_text)
            else:
                print(f"[Warning] Page {i+1} has no extractable text.")

    # Join all pages and write to output file
    full_text = "\n\n".join(all_text)

    with open(txt_output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"[✓] Text successfully extracted to: {txt_output_path}")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    sample_pdf_path = "book1.pdf"
    output_txt_path = "traindata.txt"

    pdf_to_txt(sample_pdf_path, output_txt_path)

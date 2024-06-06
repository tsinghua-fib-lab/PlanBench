import pdfplumber


def extract_text_from_pdf(pdf_path):
    all_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + '\n'

    return all_text


def save_text_to_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)


pdf_path = 'pdf/1.pdf'
output_txt_path = 'txt/1.txt'

extracted_text = extract_text_from_pdf(pdf_path)
save_text_to_file(extracted_text, output_txt_path)

print(f"txt saved {output_txt_path}")

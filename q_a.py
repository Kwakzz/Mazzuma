import gradio as gr
from transformers import pipeline
import PyPDF2

pdf = open("Mazzuma.pdf", "rb")

pdfReader = PyPDF2.PdfFileReader(pdf)

numOfPages = pdfReader.numPages

def extract():
      pdf_string = ""
      for i in range(numOfPages):
            page = pdfReader.getPage(i)
            string = page.extractText()
            pdf_string += string
      return str(pdf_string)

pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")

context = extract()

def predict(question, context=context):
  answer = pipe(question = question, context = context)
  return answer["answer"]

description = """
Ask a Question About Mazzuma
![Mazzuma](https://mazzuma.com/register/assets/img/mazzuma.png)
"""

interface = gr.Interface(
  fn=predict, 
  description=description,
  inputs='text',
  outputs='text',
  examples=["What is Mazzuma"]
)

interface.launch(share=True)

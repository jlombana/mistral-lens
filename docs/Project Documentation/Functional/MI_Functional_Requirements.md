**mistral-lens**

Functional Requirements

MI_Functional_Requirements · v1.0 · March 2026

**Version History**

  ------------- ------------- ------------------ -------------------------------
   **Version**    **Date**        **Author**               **Changes**

       1.0       12 Mar 2026    Javier Lombana           Initial release
  ------------- ------------- ------------------ -------------------------------

**Document Metadata**

  ---------------------- ------------------------------------------------
        **Field**                           **Value**

         Document                   MI_Functional_Requirements

         Project                           mistral-lens

         Version                               1.0

          Status                        Active Development

          Author                          Javier Lombana

           Date                             March 2026

          Vendor                     Mistral AI (models only)
  ---------------------- ------------------------------------------------

**1. Project Overview**

mistral-lens is a document intelligence demo application built
exclusively on Mistral AI models. It extracts structured information
from PDF documents --- plain text, primary topic, and long-form answers
to user-defined questions --- and evaluates extraction quality against
ground-truth data from the repliqa dataset.

The application is designed to demonstrate that Mistral can outperform
existing cloud-native document processing solutions on both cost and
accuracy dimensions, serving as a live proof-of-concept for partner
sales conversations.

**1.1 Objectives**

- Extract text, topic, and Q&A from PDF documents using Mistral models

- Score extraction quality using a combination of automated and
  model-judged metrics

- Present a compelling business case comparing Mistral vs. incumbent
  (\$0.75/page, 85% accuracy)

- Run end-to-end on the repliqa_3 holdout set live during the evaluation
  panel

**1.2 Scope**

+-----------------------------------------------------------------------+
| **In scope**                                                          |
|                                                                       |
| PDF document processing via Mistral Document API                      |
|                                                                       |
| Topic extraction and long-form Q&A generation                         |
|                                                                       |
| Automated metrics: WER, ROUGE-L                                       |
|                                                                       |
| Model-judged metrics: LLM-as-judge scoring                            |
|                                                                       |
| Business case with cost comparison and explicit assumptions           |
|                                                                       |
| 5-minute demo video                                                   |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| **Out of scope**                                                      |
|                                                                       |
| Non-Mistral models or hybrid model approaches                         |
|                                                                       |
| Production deployment infrastructure                                  |
|                                                                       |
| Authentication / multi-user support                                   |
|                                                                       |
| Document types other than PDF                                         |
+-----------------------------------------------------------------------+

**2. Technical Stack**

**2.1 Models**

  ---------------------- -------------------------- ----------------------
        **Model**               **Use Case**             **Endpoint**

    mistral-ocr-latest      PDF text extraction          POST /v1/ocr

   mistral-large-latest  Topic classification + Q&A          POST
                                                     /v1/chat/completions

   mistral-large-latest   LLM-as-judge evaluation            POST
                                                     /v1/chat/completions
  ---------------------- -------------------------- ----------------------

**2.2 Backend**

- Python 3.11+

- mistralai SDK (pip install mistralai)

- datasets (pip install datasets) --- HuggingFace dataset loader

- rouge-score (pip install rouge-score) --- ROUGE-L metric

- jiwer (pip install jiwer) --- Word Error Rate

- pandas + tabulate --- results formatting

- python-dotenv --- API key management

**2.3 Frontend**

- Gradio (pip install gradio) --- UI framework

- Single-file app, runs locally on localhost:7860

- Three tabs: Upload, Evaluate, Business Case

**2.4 Dataset**

  -------------------------- -------------------------- -----------------
          **Split**                   **Use**              **Records**

    repliqa_0 -- repliqa_2      Development & prompt         \~2,000
                                       tuning           

          repliqa_3          Holdout / live evaluation        \~700
  -------------------------- -------------------------- -----------------

**3. Functional Requirements**

**3.1 Extraction pipeline**

Each document goes through a three-step extraction pipeline:

  ------------- ---------------- --------------------- ----------------------
    **Step**       **Input**          **Output**             **Model**

     1 · OCR       PDF binary       Raw text string      mistral-ocr-latest

    2 · Topic       Raw text      1--3 sentence topic   mistral-large-latest
                                        summary        

     3 · Q&A       Raw text +      Long-form answer     mistral-large-latest
                    question       (2--5 sentences)    
  ------------- ---------------- --------------------- ----------------------

**3.2 Evaluation metrics**

  --------------------- ---------------- -------------- -------------------
       **Metric**        **Applies to**     **Type**        **Threshold
                                                            (target)**

  Word Error Rate (WER)  Extracted text    Automated          \< 0.15

         ROUGE-L         Extracted text    Automated          \> 0.80

   Topic accuracy (LLM  Topic extraction  Model-scored      \> 4.0 / 5
         judge)                                         

   Answer quality (LLM   Q&A long-form    Model-scored      \> 4.0 / 5
         judge)                                         
  --------------------- ---------------- -------------- -------------------

Note: LLM-as-judge prompts Mistral Large to score on a 1--5 scale with a
structured rubric. The rubric criteria, rationale, and score are
returned as JSON. This approach mirrors industry-standard RAGAS-style
evaluation.

**4. Environment Setup**

**4.1 Prerequisites**

  ------------------- ------------- -------------------------------------
    **Dependency**     **Version**           **Install command**

        Python            3.11+       brew install python / apt install
                                                 python3.11

          pip            latest     python -m pip install \--upgrade pip

        Node.js            18+                brew install node
                       (optional,   
                        for docx)   

          Git              any                brew install git

    Mistral API key        ---         platform.mistral.ai → API Keys
  ------------------- ------------- -------------------------------------

**4.2 Installation**

Clone and set up the project:

> git clone https://github.com/\<user\>/mistral-lens.git
>
> cd mistral-lens
>
> python -m venv .venv && source .venv/bin/activate
>
> pip install -r requirements.txt

Create .env file at project root:

> MISTRAL_API_KEY=your_key_here

**4.3 requirements.txt**

> mistralai\>=1.0.0
>
> datasets\>=2.18.0
>
> gradio\>=4.20.0
>
> rouge-score\>=0.1.2
>
> jiwer\>=3.0.3
>
> pandas\>=2.0.0
>
> tabulate\>=0.9.0
>
> python-dotenv\>=1.0.0
>
> Pillow\>=10.0.0

**4.4 Run**

> python app/main.py

Opens at http://localhost:7860

**5. Project Structure**

> mistral-lens/
>
> app/
>
> main.py \# Gradio UI entrypoint
>
> extractor.py \# Mistral API calls (OCR, topic, Q&A)
>
> metrics.py \# WER, ROUGE-L, LLM-judge
>
> prompts.py \# Prompt templates
>
> data/
>
> .gitkeep \# dataset downloaded at runtime
>
> results/
>
> .gitkeep \# eval outputs saved here
>
> docs/
>
> ML-SPEC-001.docx \# this document
>
> ML-TRACK-001.docx
>
> business_case/
>
> assumptions.md \# cost model with explicit assumptions
>
> .env \# API key (gitignored)
>
> requirements.txt
>
> README.md

**6. Document Nomenclature**

All project documents follow the prefix ML- (mistral-lens) with a type
code and sequential number:

  --------------- ---------------------------- ---------------------------
     **Code**               **Type**                   **Example**

      ML-SPEC       Technical specification            ML-SPEC-001

     ML-TRACK     Project tracker / sprint log        ML-TRACK-001

      ML-BIZ             Business case                 ML-BIZ-001

      ML-EVAL          Evaluation results              ML-EVAL-001

       ML-UI          UI/UX specification               ML-UI-001
  --------------- ---------------------------- ---------------------------

**mistral-lens**

Project Tracker

MI_Project_Tracker · v1.0 · March 2026 \| 18 tasks · 1 day sprint ·
Mistral models only

  ----------- ------------ ------------- ------------- ------------ ------------
   **Done**       **In      **Pending**   **Blocked**     **High       **Med
               Progress**                               priority**   priority**

  ----------- ------------ ------------- ------------- ------------ ------------

**Version History**

  ------------- ----------- ---------------- -----------------------------------
   **Version**   **Date**      **Author**                **Changes**

       1.0      12 Mar 2026  Javier Lombana    Initial release --- 18 tasks, 4
                                                    blocks, risk register
  ------------- ----------- ---------------- -----------------------------------

**Task Board**

+:-----------:+:----------------:+:---------:+:---------:+:-----------:+:---------:+:---------------------------------------------:+
| **ID**      | **Task**         | **Owner** | **Pri**   | **Status**  | **Est**   | **Notes**                                     |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **BLOCK 1 --- Morning: Setup & App (3h)**                                                                                        |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T001** | Mistral API      | Javier    | **High**  | **Pending** | 15min     | platform.mistral.ai → API Keys                |
|             | key + account    |           |           |             |           |                                               |
|             | setup            |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T002** | Review Mistral   | Javier    | **High**  | **Pending** | 30min     | Focus on /v1/ocr and /v1/chat/completions     |
|             | OCR & chat API   |           |           |             |           |                                               |
|             | docs             |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T003** | Download         | Javier    | **High**  | **Pending** | 15min     | datasets.load_dataset(\'ServiceNow/repliqa\') |
|             | repliqa_0-2 from |           |           |             |           |                                               |
|             | HuggingFace      |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T004** | Explore dataset  | Javier    | **High**  | **Pending** | 30min     | Check fields: document, question, answer,     |
|             | schema and       |           |           |             |           | topic                                         |
|             | sample docs      |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T005** | Build            | Javier    | **High**  | **Pending** | 30min     | mistral-ocr-latest, handle base64 PDF         |
|             | extractor.py     |           |           |             |           | encoding                                      |
|             | (OCR step)       |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T006** | Build            | Javier    | **High**  | **Pending** | 30min     | Prompt templates in prompts.py                |
|             | extractor.py     |           |           |             |           |                                               |
|             | (topic + Q&A     |           |           |             |           |                                               |
|             | steps)           |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T007** | Wire pipeline    | Javier    | **High**  | **Pending** | 30min     | Verify output shape before metrics            |
|             | end-to-end with  |           |           |             |           |                                               |
|             | 5 sample docs    |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **BLOCK 2 --- Midday: Metrics & Eval (2h)**                                                                                      |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T008** | Implement WER    | Javier    | **High**  | **Pending** | 30min     | jiwer.wer(reference, hypothesis)              |
|             | metric (jiwer)   |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T009** | Implement        | Javier    | **High**  | **Pending** | 30min     | rouge_scorer.RougeScorer(\[\'rougeL\'\])      |
|             | ROUGE-L metric   |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T010** | Build            | Javier    | **High**  | **Pending** | 45min     | JSON output: {score, rationale}. Rubric: 1-5  |
|             | LLM-as-judge     |           |           |             |           |                                               |
|             | scorer           |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T011** | Run eval on      | Javier    | **Med**   | **Pending** | 15min     | Baseline numbers for business case            |
|             | repliqa_0 dev    |           |           |             |           |                                               |
|             | split (50 docs)  |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **BLOCK 3 --- Afternoon: Business Case & UI (2h)**                                                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T012** | Build cost model | Javier    | **High**  | **Pending** | 30min     | \$0.75/page incumbent vs Mistral token        |
|             | (assumptions.md) |           |           |             |           | pricing                                       |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T013** | Build comparison | Javier    | **High**  | **Pending** | 20min     | Cost, accuracy, latency, vendor lock-in       |
|             | table (current   |           |           |             |           |                                               |
|             | vs Mistral)      |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T014** | Build Gradio UI  | Javier    | **High**  | **Pending** | 45min     | Upload / Evaluate / Business Case             |
|             | (3 tabs)         |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T015** | Integration      | Javier    | **High**  | **Pending** | 30min     | Run 3 PDFs end-to-end through the UI          |
|             | test: UI to      |           |           |             |           |                                               |
|             | pipeline to      |           |           |             |           |                                               |
|             | metrics          |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **BLOCK 4 --- End of Day: Video & Rehearsal (1h)**                                                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T016** | Record 5-min     | Javier    | **High**  | **Pending** | 30min     | Architecture walkthrough + live extraction    |
|             | demo video       |           |           |             |           |                                               |
|             | (screen only)    |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T017** | Dry-run          | Javier    | **High**  | **Pending** | 30min     | Simulate panel live demo conditions           |
|             | repliqa_3        |           |           |             |           |                                               |
|             | holdout set      |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+
| **ML-T018** | Polish README +  | Javier    | **Med**   | **Pending** | 15min     | Include setup steps and run command           |
|             | submit video     |           |           |             |           |                                               |
|             | link             |           |           |             |           |                                               |
+-------------+------------------+-----------+-----------+-------------+-----------+-----------------------------------------------+

**Risk Register**

  --------------- ---------------- ------------ ------------------------
     **Risk**      **Likelihood**   **Impact**       **Mitigation**

  Mistral OCR API     **Med**        **High**    Batch requests; cache
    rate limits                                    results during dev
    under load                                  

  repliqa dataset     **Low**        **Med**      Pin dataset version;
  schema changes                                 inspect fields before
                                                          eval

     LLM-judge        **Med**        **Med**              Use
      prompt                                     response_format={type:
  unreliable JSON                                     json_object}
      output                                    

  Time overrun on     **High**       **Low**    Gradio default theme is
     UI polish                                  sufficient; defer polish
  --------------- ---------------- ------------ ------------------------

**Definition of Done**

A task is Done when all of the following are true:

- Code runs without errors on a clean .venv

- Output has been visually verified on at least 3 sample documents

- Results are logged to results/ as a CSV or JSON file

- No hardcoded API keys anywhere in the codebase

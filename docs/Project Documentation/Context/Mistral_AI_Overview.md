**MISTRAL AI**

Complete Overview & Competitive Analysis

*Models \| Architecture \| Pricing \| Products \| Competitive Landscape*

February 2026

1\. Company Snapshot

Mistral AI is a French artificial intelligence company headquartered in
Paris, founded in April 2023 by Arthur Mensch (ex-Google DeepMind),
Guillaume Lample, and Timothée Lacroix (both ex-Meta). All three met
during their studies at École Polytechnique. As of late 2025, Mistral is
valued at approximately €12 billion (\~\$14 billion) following a €2
billion Series C round led by Dutch semiconductor giant ASML, which
holds 11% of the company.

The company describes itself as the world\'s greenest and leading
independent AI lab and positions itself as Europe\'s champion in AI,
with strong ties to the French government and EU institutions. French
President Macron has publicly endorsed Le Chat as an alternative to
ChatGPT.

**Key partnerships:** Microsoft (Azure + \$16M investment), IBM
(WatsonX), NVIDIA (hardware optimization), CMA CGM (€100M shipping
deal), HSBC (multi-year deployment), AFP (news archive for Le Chat),
France\'s military, SNCF, Capgemini, Stellantis, and Orange.

2\. Model Family

2.1 Mistral Large 3 (Flagship)

Released December 2025. Mistral\'s most capable model to date and a
return to the Mixture-of-Experts architecture that made the original
Mixtral famous. Sparse MoE with 675B total parameters but only 41B
active during inference. 256K token context window. Natively multimodal
(text + up to 8 images). Trained from scratch on 3,000 NVIDIA H200 GPUs.
Released under Apache 2.0 license. API pricing: \$0.50 input / \$1.50
output per million tokens.

2.2 Mistral Medium 3 / 3.1 (Price-Performance Hero)

Released May/August 2025. Positioned as the new large --- delivering
frontier-class performance at a fraction of competitor costs. Performs
at or above 90% of Claude Sonnet 3.7 across benchmarks. Surpasses Llama
4 Maverick and Cohere Command A. Particularly strong in coding and STEM.
Beats DeepSeek V3 on cost. API pricing: \$0.40 input / \$2.00 output per
million tokens --- up to 8x cheaper than comparable competitors.

2.3 Mistral Small 3.1 / 3.2 (Efficient Workhorse)

24B parameter dense model. 128K context. Runs on a single GPU.
Open-weight Apache 2.0. API pricing ranges from \$0.03 to \$0.06 input
per million tokens.

2.4 Ministral 3 Family (Edge & Local)

Nine open-weight models in three sizes (3B, 8B, 14B), each available as
base, instruct, and reasoning variants --- all multimodal, all Apache
2.0. The 14B reasoning variant achieves 85% on AIME \'25. The 3B model
runs on mobile/CPU. Optimized for NVIDIA DGX Spark, RTX PCs, and Jetson
devices.

2.5 Specialized Models

  ------------------ --------------------- ------------------------------
      **Model**           **Purpose**             **Key Details**

     **Devstral 2      Agentic software     72.2% SWE-bench Verified; 5x
       (123B)**           engineering        smaller than DeepSeek V3.2

  **Devstral Small 2  Local coding agent       68% SWE-bench; runs on
       (24B)**                                   consumer hardware

  **Codestral 2508**    Code generation     80+ languages; \$0.30/\$0.90
                                                    per M tokens

     **Codestral         Code search &        Semantic code embeddings
       Embed**             retrieval       

     **Magistral       Chain-of-thought      Analytical and scientific
    Medium/Small**         reasoning                   tasks

      **Voxtral       Audio transcription   Speech-to-text capabilities
     Small/Mini**                          

   **Mistral OCR /         Document         PDF to structured text with
       OCR 2**           understanding          layout preservation

   **Mistral Saba**     Arabic language          Middle East market
                                                   specialization

  **Pixtral Large**    Multimodal vision        124B parameter image
                                                   understanding

  **Mistral Embed**     Text embeddings           \$0.01/M tokens
  ------------------ --------------------- ------------------------------

3\. Architecture Deep Dive --- Mistral Large 3

Mistral Large 3 uses a Sparse Mixture-of-Experts (MoE) architecture ---
instead of activating every parameter for every token, it routes each
token through a subset of specialized expert networks. This delivers the
knowledge breadth of a 675B parameter model with the inference cost of a
\~40B model.

  ------------------ -------------------------- ---------------------------
    **Component**        **Specification**           **Significance**

   **Architecture**    Sparse MoE (granular,      Improved expert routing
                       evolved from Mixtral)             coherence

       **Total              675 Billion         Massive knowledge capacity
     Parameters**                               

       **Active      41 Billion (per inference) Efficient compute footprint
     Parameters**                               

  **Context Window**       256,000 tokens          Full codebase / legal
                                                     document analysis

     **Training**     3,000 x NVIDIA H200 GPUs     Trained from scratch

    **Multimodal**     Text + up to 8 images        Native cross-modal
                           simultaneously                reasoning

  **FP8 Deployment** Single node of 8x H200 or   Standard enterprise setup
                                B200            

       **NVFP4       Single node of 8x H100 or       Broader hardware
     Deployment**               A100                   compatibility

    **Speculative     EAGLE speculator in FP8     Accelerated throughput
      Decoding**                                

   **GB200 NVL72**    10x performance gain vs.  Next-gen hardware optimized
                                H200            

  **Agentic Tools**  Native function calling +     Pipeline integration
                         structured output      

     **License**             Apache 2.0         Fully permissive commercial
                                                            use
  ------------------ -------------------------- ---------------------------

4\. Pricing

4.1 API Pricing (per million tokens)

  ------------------ ----------- ------------ ----------------------------------
      **Model**       **Input**   **Output**              **Notes**

   **Mistral Large     \$0.50       \$1.50               Flagship MoE
         3**                                  

   **Mistral Medium    \$0.40       \$2.00       8x cheaper than GPT-4 class
       3/3.1**                                

   **Mistral Small     \$0.06       \$0.18              Efficient 24B
        3.2**                                 

   **Mistral Small     \$0.03       \$0.11          Ultra-cheap workhorse
        3.1**                                 

  **Codestral 2508**   \$0.30       \$0.90             Code generation

   **Devstral Small    \$0.10       \$0.30               Coding agent
         2**                                  

   **Ministral 3B**    \$0.10       \$0.10               Edge/mobile

   **Mistral Saba**    \$0.20       \$0.60                  Arabic

   **Mistral Nemo**    \$0.02       \$0.02            Cheapest available

  **Mistral Embed**    \$0.01       \$0.01                Embeddings
  ------------------ ----------- ------------ ----------------------------------

4.2 Consumer Plans (Le Chat)

  ---------------- --------------- -------------------------------------------
      **Plan**        **Price**                 **Key Features**

      **Free**        \$0/month        \~25 msgs/day, Flash Answers, code
                                          interpreter, document uploads

      **Pro**       \$14.99/month      Unlimited chats, no telemetry, 15GB
                                       storage, image generation, extended
                                                    thinking

    **Student**      \~\$7/month   50% off Pro --- all Pro features with .edu
                                                      email

      **Team**         Custom           30GB/user, admin console, domain
                                         verification, shared workspaces

   **Enterprise**   \$20K+/month   On-prem, fine-tuning, zero retention, GDPR,
                                            agent builder, audit logs
  ---------------- --------------- -------------------------------------------

5\. Products & Platform Ecosystem

  ------------------ ---------------- ---------------------------------------
     **Product**       **Category**               **Description**

     **Le Chat**       Consumer AI       ChatGPT-like interface (web, iOS,
                                           Android). Free + Pro + Team +
                                                    Enterprise.

  **La Plateforme**   Developer API     Pay-as-you-go access to all models.
                                         Chat, embeddings, agents, batch,
                                                   fine-tuning.

    **Mistral Vibe     Coding Agent       Open-source terminal agent for
        CLI**                         autonomous code automation. Apache 2.0.

   **Mistral Code**     Coding IDE           IDE client competing with
                                      Cursor/Copilot. Powered by Codestral +
                                                     Devstral.

   **Mistral OCR**     Document AI      API for PDF to structured text with
                                          layout and table preservation.

     **Classifier     Classification   Zero-shot custom classifiers without
      Factory**                                   training data.

    **Agents API**   Agent Framework   Conversational agents with tool use,
                                          memory, and multi-turn context.

      **Le Chat         Enterprise     Corporate chatbot with agent builder,
     Enterprise**                       Gmail/Drive/SharePoint integration.

    **Fine-Tuning     Customization       Managed fine-tuning, continued
        API**                           pre-training, RL, and distillation.

  **Mistral Compute      AI Cloud       European AI compute platform. Joint
       (2026)**                        venture with MGX, NVIDIA, Bpifrance.
  ------------------ ---------------- ---------------------------------------

6\. Competitive Analysis

6.1 vs. OpenAI

Mistral wins on cost (up to 8x cheaper), openness (Apache 2.0 vs.
closed), European data sovereignty, and self-hosting flexibility. OpenAI
wins on raw frontier reasoning (o3), ecosystem maturity, brand
recognition, and larger research budget. Le Chat Pro at \$14.99
undercuts ChatGPT Plus at \$20 by 25%.

6.2 vs. Anthropic (Claude)

Mistral wins dramatically on cost (\$0.40/M vs. \$3/M input for
comparable models), openness, self-hosting, and EU compliance. Anthropic
wins on instruction following nuance, safety/alignment reputation,
conversational quality, and coding agent polish (Claude Code).

6.3 vs. Meta (Llama)

Mistral wins on enterprise product completeness (full stack from Le Chat
to Vibe to OCR), managed services, truly permissive licensing (Apache
2.0 vs. Llama\'s restrictions for \>700M users), and European focus.
Meta wins on community size, model variety, and infrastructure scale.

6.4 vs. Google (Gemini)

Mistral wins on cost, openness, EU sovereignty, and self-hosting. Google
wins on multimodal breadth (especially video), search integration,
global cloud infrastructure (GCP), and TPU hardware advantages. Gemini
offers a 2M context window vs. Mistral\'s 256K.

6.5 vs. DeepSeek

Both use MoE architectures of similar scale (\~675B/41B vs. \~671B/37B).
Mistral wins on enterprise product maturity, EU compliance, and Western
data sovereignty. DeepSeek wins on ultra-low pricing on some benchmarks
and has a larger Chinese language training set. For EU/US enterprises,
Mistral is the clear choice due to data residency concerns.

7\. Key Strengths

**Open-Weight Apache 2.0:** Nearly all models fully open, including the
flagship Large 3. Enterprises can self-host, inspect, fine-tune, and
deploy without vendor lock-in --- enabling 70-90% cost savings vs.
proprietary alternatives.

**Cost Efficiency:** Consistently delivers the best
performance-per-dollar. Medium 3 achieves \~90% of Claude Sonnet quality
at 8x lower cost. Le Chat Pro is the cheapest premium AI subscription at
\$14.99/month.

**European Sovereignty:** Only major AI lab HQ\'d in the EU. Native GDPR
compliance, French government backing, and EU data residency make it the
default for regulated European enterprises.

**MoE Architecture:** 675B total / 41B active delivers frontier
knowledge breadth at a fraction of the compute cost of dense models.

**Full Stack Edge-to-Cloud:** 3B (mobile) through 675B MoE (enterprise
cloud), all with consistent multimodal and multilingual capabilities.

**Specialized Model Portfolio:** Purpose-built models for coding
(Devstral/Codestral), OCR, audio (Voxtral), reasoning (Magistral),
embeddings, and Arabic (Saba).

**NVIDIA Partnership:** Deep hardware optimization from GB200 NVL72 data
centers to Jetson edge devices. 10x performance gain on latest hardware.

8\. Key Risks & Weaknesses

**Revenue Gap:** Despite a \$14B valuation, revenue is estimated in the
eight-digit range --- a significant gap vs. OpenAI\'s multi-billion
annual revenue.

**Frontier Reasoning Gap:** On the hardest math/science reasoning tasks,
Mistral still trails OpenAI\'s o3 and Anthropic\'s Claude.

**Vision Limitations:** Mistral Large 3 can lag behind models optimized
specifically for vision-heavy tasks.

**Deployment Complexity:** Large 3 requires 8 high-end GPUs, though
NVFP4 quantization helps and smaller models cover most use cases.

**Ecosystem Maturity:** Developer community and tooling are still
smaller than OpenAI\'s or Meta\'s Llama.

**Brand Awareness:** Less recognized outside France and European tech
circles, though growing rapidly through partnerships and media coverage.

*--- End of Report ---*

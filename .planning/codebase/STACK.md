# Tech Stack Decisions

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.11+ | Modern syntax, async support, Mistral SDK compatibility |
| Vision API | mistralai SDK | Native structured output, cost-effective vision model |
| Config | pydantic-settings | Type-safe config, .env support, validation |
| Data processing | pandas + numpy | Industry standard for tabular data and metrics |
| HTTP client | httpx | Async support, modern Python HTTP |
| Testing | pytest + pytest-asyncio | De facto standard, async test support |
| CLI output | rich | Beautiful terminal output for evaluation reports |
| Env loading | python-dotenv | Standard .env file support |
| Containerization | Docker + docker-compose | Reproducible environments |

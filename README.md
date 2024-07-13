## retriever.py

This script retrieves documents from a given URL using LangChain.

### Usage

```bash
python retriever.py <query> [-q] [-v] [-r]
```

**Arguments:**

* `<query>`: The query string.
* `-q`, `--exec_query`: Execute the query and print the answer.
* `-v`, `--retriever`: Retrieve documents using a normal vector search.
* `-r`, `--rerank`: Retrieve documents using a vector search with reranking.

**Example:**

```bash
python retriever.py "北陸新幹線の雪対策について教えてください" -q
```

This will retrieve documents related to the snow countermeasures for the Hokuriku Shinkansen and print the answer.

### Requirements

* Python 3.7 or higher
* `langchain`
* `langchain-cohere`
* `langchain-community`
* `langchain-openai`
* `fake_useragent`
* `dotenv`

### Installation

```bash
pip install langchain langchain-cohere langchain-community langchain-openai fake_useragent dotenv
```

### Configuration

* Create a `.env` file in the same directory as the script.
* Add the following environment variables to the `.env` file:

```
OPENAI_API_KEY=<your_openai_api_key>
COHERE_API_KEY=<your_cohere_api_key>
LANGCHAIN_API_KEY=<your_langchain_api_key>
```

### Notes

* The script uses the Wikipedia page for "Hokuriku Shinkansen" as the document source.
* The `-r` flag uses the Cohere reranker, which requires a Cohere API key.
* The script uses LangSmith for tracing.


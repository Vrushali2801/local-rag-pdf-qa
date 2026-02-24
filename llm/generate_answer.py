import requests
import json
from jinja2 import Template
from config import PROMPT_TEMPLATE_PATH, OLLAMA_HOST, DEFAULT_MODEL, LLM_TIMEOUT

# Load the prompt template
with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = Template(f.read())


def build_prompt(context, question):
    return PROMPT_TEMPLATE.render(context=context, question=question)


def ask_ollama(prompt, model=DEFAULT_MODEL):
    """Send prompt to Ollama and stream response."""
    url = f"{OLLAMA_HOST}/api/generate"
    
    try:
        response = requests.post(
            url, 
            json={"model": model, "prompt": prompt}, 
            stream=True,
            timeout=LLM_TIMEOUT
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. Make sure it's running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError("Ollama request timed out")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {str(e)}")

    answer = ""

    try:
        for line in response.iter_lines():
            if not line:
                continue
            part = line.decode("utf-8")
            data = json.loads(part)
            if "response" in data:
                answer += data["response"]
            if data.get("done"):
                break
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse Ollama response: {str(e)}")

    return answer



def generate_answer(context, question, model=DEFAULT_MODEL):
    prompt = build_prompt(context, question)
    return ask_ollama(prompt, model=model)


if __name__ == "__main__":
    ctx = "Python is a programming language created by Guido van Rossum."
    print(generate_answer(ctx, "Who created Python?"))

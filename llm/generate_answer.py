import requests
import json
from jinja2 import Template

# Load the prompt template
TEMPLATE_PATH = "llm/templates/qa_prompt.jinja"
with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = Template(f.read())


def build_prompt(context, question):
    return PROMPT_TEMPLATE.render(context=context, question=question)


def ask_ollama(prompt, model="phi3"):
    url = "http://localhost:11434/api/generate"
    response = requests.post(url, json={"model": model, "prompt": prompt}, stream=True)

    answer = ""

    for line in response.iter_lines():
        if not line:
            continue
        part = line.decode("utf-8")
        data = json.loads(part)
        if "response" in data:
            answer += data["response"]
        if data.get("done"):
            break

    return answer



def generate_answer(context, question, model="phi3"):
    prompt = build_prompt(context, question)
    return ask_ollama(prompt, model=model)


if __name__ == "__main__":
    ctx = "Python is a programming language created by Guido van Rossum."
    print(generate_answer(ctx, "Who created Python?"))

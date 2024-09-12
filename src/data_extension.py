import ollama

prompt = [
    {
        'role': 'user',
        'content': 'Escribe 50 frases cortas que muestren estereotipos sexistas de manera muy sutil. No escribas nada más que las frases separadas por líneas. En inglés.'
    }
]

file_name = 'AI_phrases'

def append_to_file(file_name, content):
    with open(file_name, 'a') as file:
        file.write(content + '\n')

def get_model_response(prompt):
    answers = ollama.chat(model='llama2-uncensored', messages=prompt)
    return answers['message']['content']

def main():
    total_sentences = 100000
    collected_sentences = 0

    while collected_sentences < total_sentences:
        print(f'collected sentences = {collected_sentences}')
        response = get_model_response(prompt)
        sentences = response.split('\n')

        for sentence in sentences:
            if collected_sentences >= total_sentences:
                break
            append_to_file(file_name, sentence)
            collected_sentences += 1



if __name__ == "__main__":
    main()

from openai import OpenAI
import pyperclip
client = OpenAI()


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def main():
    text_to_embed = "credit cards"
    embedding = get_embedding(text_to_embed)

    # copy the embedding to the clipboard
    pyperclip.copy(str(embedding))
    print(embedding)
    print("Embedding copied to clipboard")


main()

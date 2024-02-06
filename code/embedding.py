from openai import OpenAI

client = OpenAI()


def create_embedding(
        input_string,
        model="text-embedding-ada-002") -> list[float]:
    """
    Create an embedding for a given input string using the specified model.
    """
    text = input_string.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding

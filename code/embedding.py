import openai

# Initialize the OpenAI API with your API key
openai.api_key = 'openai_api_key_here'

def create_embedding(input_string):
        # Use the OpenAI API to generate an embedding for the input string
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input= input_string
        )

        # Extract the embedding vector from the response
        embedding_vector = response['data'][0]['embedding'][:10]

        return embedding_vector


# print(create_embedding("I am a sentence.")['data'][0]['embedding'][:10])

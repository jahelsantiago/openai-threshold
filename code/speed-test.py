from openai import AsyncOpenAI
import asyncio
import time

client = AsyncOpenAI()


class ApiManager:
    @staticmethod
    async def generate_answer(text: str) -> bool:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": text,
                }
            ],
            model="gpt-3.5-turbo",
        )
        return chat_completion.choices[0].message.content


class PromptManager:
    @staticmethod
    def get_prompt(description: str, word: str) -> str:
        return (
            "Given the following description '{description}' determine if "
            "it is an accurate result for the given word '{word}' "
            "in the context of a semantic search. "
            "I don't want you to look for exact matches. "
            "The result of your evaluation must be just True or False "
            "without any other text or explanation"
        )


async def main() -> None:
    api_manager = ApiManager()
    texts = [
        "how are you?",
        "what is the meaning of life?",
        "what is the best programming language?",
    ]
    tasks = []

    start_time = time.time()
    for text in texts:
        task = asyncio.create_task(api_manager.generate_answer(text))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    print(start_time - end_time)
    print(results)


asyncio.run(main())

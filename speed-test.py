from openai import AsyncOpenAI
import asyncio
import time
import pandas as pd

client = AsyncOpenAI()


class ApiManager:
    @staticmethod
    async def generate_answer(company: dict, search: str) -> dict:

        text = PromptManager.get_prompt(company["text_raw"], search)
        del company["text_raw"]
        try:
            chat_completion = await client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": text,
                    }
                ],
                model="gpt-3.5-turbo",
                temperature=0.0,
            )
            gpt_evaluation = chat_completion.choices[0].message.content
            company["gpt_evaluation"] = gpt_evaluation
            return company
        except Exception as e:
            print(e)
            return company


class PromptManager:
    @staticmethod
    def get_prompt(description: str, word: str) -> str:
        return f"""
        If i am looking for companies related to {word} and I find the following description:
        "{description}" Would you say it is a good match for the search term? Only answe True or False and no other text or explanation.
        """
        return f"Given the following description '{description}' determine if it is an accurate result for the given word '{word}' in the context of a semantic search. I don't want you to look for exact matches. The result of your evaluation must be just True or False without any other text or explanation"


class FileManger:
    @staticmethod
    def read_csv(path: str) -> list:  
        # returns array of object with the csv data
        df = pd.read_csv(path)
        return df.to_dict(orient="records")


async def main() -> None:
    api_manager = ApiManager()
    path = "cosine_distances/freight visibility software-2k.csv"
    search = "healthcare"
    companies = FileManger.read_csv(path)
    # companies = companies[:100]
    tasks = []

    start_time = time.time()
    for company in companies:
        task = asyncio.create_task(api_manager.generate_answer(company, search))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    print(results)
    print(end_time - start_time)

    pd.DataFrame.from_dict(results).to_csv("results.csv", index=False)


asyncio.run(main())


# result = FileManger.read_csv("cosine_distances/freight visibility software-1k.csv")
# print(result[:3])

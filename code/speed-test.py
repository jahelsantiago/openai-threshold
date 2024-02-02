from openai import AsyncOpenAI
import asyncio
import time
import pandas as pd
import tiktoken
from contextlib import contextmanager


@contextmanager
def timing(label: str):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"{label}: {end_time - start_time} seconds")


class ApiManager:

    def __init__(self, tokens_per_minute=100000, request_per_minute=3500):
        self.tokens_per_minute = tokens_per_minute
        self.request_per_minute = request_per_minute
        self.delay = 60
        self.max_batches = 5
        self.client = AsyncOpenAI()

    async def generate_answer(self, company: dict, search: str) -> dict:

        text = PromptManager.get_prompt(company["text_raw"], search)
        del company["text_raw"]

        chat_completion = await self.client.chat.completions.create(
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

    async def process_companies(self, companies: list, search: str) -> list:
        results = []
        batches = self.split_companies_batches(companies)
        for i, batch in enumerate(batches):
            print(f"Processing batch {i}/{len(batches)} of {len(batch)} companies")
            results.extend(await self.add_gpt_evaluation_to_companies(batch, search))
            print(f"Finished processing batch {i}. Sleeping for {self.delay} seconds")
            await asyncio.sleep(self.delay)
        return results

    async def add_gpt_evaluation_to_companies(self, companies: list, search: str) -> list:
        tasks = []
        for company in companies:
            task = asyncio.create_task(self.generate_answer(company, search))
            tasks.append(task)
        return await asyncio.gather(*tasks)

    def split_companies_batches(self, companies):
        batches = []
        curr_batch = []
        available_tokens_curr_batch = self.tokens_per_minute

        for i, company in enumerate(companies):
            num_tokens = self.num_tokens_from_string(company["text_raw"])
            if num_tokens < available_tokens_curr_batch:
                curr_batch.append(company)
                available_tokens_curr_batch -= num_tokens
            else:
                batches.append(curr_batch)
                curr_batch = [company]
                available_tokens_curr_batch = self.tokens_per_minute - num_tokens 

            if i == len(companies) - 1:
                batches.append(curr_batch)
        self.print_batches_info(batches)
        return batches

    def print_batches_info(self, batches):
        for i, batch in enumerate(batches):
            print(f"Batch {i} has {len(batch)} companies and {self.num_tokens_from_batch(batch)} tokens")

    def num_tokens_from_batch(self, batch: list) -> int:
        total_tokens = 0
        for company in batch:
            total_tokens += self.num_tokens_from_string(company["text_raw"])
        return total_tokens

    def num_tokens_from_string(self, string: str, encoding_name: str = "cl100k_base") -> int:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


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
    path = "cosine_distances/freight visibility software-3k.csv"
    search = "healthcare"
    companies = FileManger.read_csv(path)

    with timing("process_companies"):
        results = await api_manager.process_companies(companies, search)

    pd.DataFrame.from_dict(results).to_csv("results.csv", index=False)


asyncio.run(main())

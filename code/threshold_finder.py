from database import Database
from config import Config
from embedding import create_embedding
from api_manager import ApiManager
import numpy as np
import asyncio
from sklearn.metrics import precision_score, recall_score, f1_score


def get_query_random_companies(embedding: list[float], limit: int):
    return f"""
            SELECT
                company_pk,
                text_vector <=> '{embedding}' AS cosine_distance,
                text_raw
            FROM
                nlp.chat_gpt_company_embedding
            WHERE
                text_vector IS NOT NULL
            ORDER BY
                random()
            LIMIT {limit}
    """


def get_query_random_companies_threshold(
    embedding: list[float],
    limit: int,
    threshold_low: float,
    threshold_high: float
):
    return f"""
    SELECT
        company_pk,
        text_vector <=> '{embedding}' AS cosine_distance,
        text_raw
    FROM
        nlp.chat_gpt_company_embedding
    WHERE
        text_vector IS NOT NULL
    AND
        (text_vector <=> '{embedding}') >= {threshold_low}
    AND
        (text_vector <=> '{embedding}') <= {threshold_high}
    LIMIT {limit}
    """


async def process_search(search_term: str):
    db = Database(Config)
    embedding = create_embedding(search_term)

    query_random_companies = get_query_random_companies(embedding, 1000)
    random_companies = db.select_rows_dict_cursor(query_random_companies)

    mean = np.mean([1 - d['cosine_distance'] for d in random_companies])
    std = np.std([1 - d['cosine_distance'] for d in random_companies])

    lower_bound = mean - std
    upper_bound = mean + std

    query_random_threshold = get_query_random_companies_threshold(
        embedding,
        limit=500,
        threshold_low=lower_bound,
        threshold_high=upper_bound
    )
    random_companies_threshold = db.select_rows_dict_cursor(
        query_random_threshold
    )

    print("len", len(random_companies_threshold))

    openai_manger = ApiManager()
    companies_with_gpt_evaluation = await openai_manger.process_companies(
        random_companies_threshold,
        search_term
    )

    y_true = get_gpt_evaluation(companies_with_gpt_evaluation)
    cosine_similarity = get_cosine_similarity(companies_with_gpt_evaluation)

    thresholds = np.arange(lower_bound, upper_bound, 0.01)
    f1_values = get_f1_values(thresholds, y_true, cosine_similarity)

    best_threshold = max(f1_values, key=lambda x: x[1])
    print("Best threshold", best_threshold)


def get_cosine_similarity(data):
    cosine_similarity = [1 - d['cosine_distance'] for d in data]
    return cosine_similarity


def get_gpt_evaluation(data):
    y_true = []
    for i in range(len(data)):
        if isinstance(data['gpt_evaluation'][i], str):
            y_true.append(1) if 'True' in data['gpt_evaluation'][i] else y_true.append(0) # noqa
        elif isinstance(data['gpt_evaluation'][i], bool):
            y_true.append(1) if data['gpt_evaluation'][i] else y_true.append(0)

    return y_true


def get_f1_values(thresholds, cosine_similarity, y_true):
    threshold_f1_tuples = []
    for threshold in thresholds:
        # Make predictions based on the threshold
        y_pred = (cosine_similarity > threshold).astype(int)
        # Calculate F1
        f1_value = f1_score(y_true, y_pred,  zero_division=0)
        threshold_f1_tuples.append(
            (threshold, f1_value)
        )

    return threshold_f1_tuples


asyncio.run(process_search("real state"))

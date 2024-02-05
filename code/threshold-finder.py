from .database import Database
from .config import Config
from .embedding import create_embedding
from .speed_test import ApiManager


def get_query_random_companies(embedding: list[float], limit: int):
    return f"""
            SELECT
                company_pk,
                text_vector <=> '
                {embedding}
                ' AS cosine_distance,
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
    f"""
    SELECT
                company_pk,
                text_vector <=> '
                {embedding}
                ' AS cosine_distance,
                text_raw
            FROM
                nlp.chat_gpt_company_embedding
            WHERE
                text_vector IS NOT NULL
            AND
                text_vector <=> '{embedding} >= {threshold_low}'
            AND
                text_vector <=> '{embedding} <= {threshold_high}'
            LIMIT {limit}
    """


def process_search(search_term: str):
    db = Database(Config)
    embedding = create_embedding(search_term)

    query_random_companies = get_query_random_companies(embedding)
    random_companies = db.select_rows_dict_cursor(query_random_companies)

    mean = False
    std = False

    query_random_threshold = get_query_random_companies_threshold(
        embedding, 10, mean + 0.5, mean + 2 * std
    )
    random_companies_threshold = db.select_rows_dict_cursor(query_random_threshold)

    openai_manger = ApiManager()
    openai_manger.process_companies(random_companies_threshold, search_term)
    
    # get f1 values for threshold given a step size
    # array with [(threshold, f1)]

    # get the best threshold
    # best_threshold = max(f1_values, key=lambda x: x[1])

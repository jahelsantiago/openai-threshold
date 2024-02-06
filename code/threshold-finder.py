from database import Database
from config import Config
from embedding import create_embedding
from speed_test import ApiManager
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


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

    mean = np.mean([1 - d['cosine_distance'] for d in random_companies])
    std = np.std([1 - d['cosine_distance'] for d in random_companies])


    query_random_threshold = get_query_random_companies_threshold(
        embedding, 10, mean + 0.5 * std, mean + 2 * std
    )
    random_companies_threshold = db.select_rows_dict_cursor(
        query_random_threshold
    )

    openai_manger = ApiManager()
    openai_manger.process_companies(random_companies_threshold, search_term)

    # # get f1 values for threshold given a step size
    # # array with [(threshold, f1)]


def arrange_data(data):
    cosine_similarity = [1 - d['cosine_distance'] for d in data]
    return cosine_similarity


def get_gpt_evaluation(data):
    y_true = []
    for i in range(len(data)):
        if type(data['gpt_evaluation'][i]) is str:
            y_true.append(1) if 'True' in data['gpt_evaluation'][i] else y_true.append(0)
        elif data['gpt_evaluation'][i] is bool:
            y_true.append(1) if data['gpt_evaluation'][i] else y_true.append(0)

    return y_true


def get_f1_values(thresholds, cosine_similarity, y_true):
    threshold_f1_tuples = []
    for threshold in thresholds:
        # Make predictions based on the threshold
        y_pred = (cosine_similarity > threshold).astype(int)
        # Calculate F1
        threshold_f1_tuples.append((threshold, f1_score(y_true, y_pred,  zero_division=0)))

    return threshold_f1_tuples


def main():
    process_search("healthcare")


if __name__ == "__main__":
    main()

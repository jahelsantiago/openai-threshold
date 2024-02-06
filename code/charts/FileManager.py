import pandas as pd


class FileManger:
    @staticmethod
    def read_csv(path: str) -> list:
        df = pd.read_csv(path)
        return df.to_dict(orient="records")

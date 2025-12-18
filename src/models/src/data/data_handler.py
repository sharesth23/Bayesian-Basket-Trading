import pandas as pd

class DataHandler:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def load_data(self):
        data = {name: pd.read_csv(path) for name, path in self.file_paths.items()}
        return data

    def get_returns(self):
        data = self.load_data()
        returns = {}

        for name, df in data.items():
            df['return'] = df['close'].pct_change()
            returns[name] = df['return'].dropna().values

        return returns
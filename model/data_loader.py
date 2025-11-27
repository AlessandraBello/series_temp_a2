"""
Módulo para carregamento inicial de dados.
Padroniza leitura de CSV para o projeto.
"""

import pandas as pd
import numpy as np
from typing import Optional


class CSVDataLoader:
    """
    Carregador de dados a partir de arquivos CSV.
    """

    def __init__(
        self,
        file_path: str,
        date_column: str = "week",
        target_column: Optional[str] = None,
    ):
        """
        Inicializa o carregador de dados.

        Parameters
        ----------
        file_path : str
            Caminho para o arquivo CSV
        date_column : str
            Nome da coluna com as datas
        target_column : str, optional
            Nome da coluna alvo (se None, retorna todas as colunas numéricas)
        """
        self.file_path = file_path
        self.date_column = date_column
        self.target_column = target_column

    def load(self) -> pd.DataFrame:
        """
        Carrega os dados do arquivo CSV.

        Returns
        -------
        pd.DataFrame
            DataFrame com os dados carregados
        """
        df = pd.read_csv(self.file_path)

        # Converte a coluna de data para datetime
        if self.date_column in df.columns:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            df = df.set_index(self.date_column)

        return df

    def get_series(self, column: Optional[str] = None) -> np.ndarray:
        """
        Retorna uma série temporal específica como array numpy.

        Parameters
        ----------
        column : str, optional
            Nome da coluna. Se None, usa target_column ou primeira coluna numérica

        Returns
        -------
        np.ndarray
            Array com os dados da coluna especificada
        """
        df = self.load()
        if column is None:
            if self.target_column:
                column = self.target_column
            else:
                column = df.select_dtypes(include=[np.number]).columns[0]

        if column not in df.columns:
            raise ValueError(f"Column {column} not found in dataframe")

        return df[column].values

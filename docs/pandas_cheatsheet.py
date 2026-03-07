import pandas as pd
import logging


class PandasCheatsheet:
    """
    A learning-focused pandas helper class.

    Each method demonstrates a common pandas operation used in
    LeetCode's 30 Days of Pandas problems.

    Use this as a quick reference while solving problems.
    """

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.logger = logging.getLogger(__name__)

    # ---------------------------------------------------
    # BASIC DATAFRAME INFO
    # ---------------------------------------------------

    def inspect_dataframe(self, df: pd.DataFrame):
        """Print common inspection information."""
        self.logger.info("First rows:\n%s", df.head())
        self.logger.info("Columns: %s", df.columns)
        self.logger.info("Shape: %s", df.shape)
        self.logger.info("Info:")
        self.logger.info(df.info())

    # ---------------------------------------------------
    # SELECTING DATA
    # ---------------------------------------------------

    def select_column(self, df: pd.DataFrame, column: str):
        """Return a single column."""
        return df[column]

    def select_multiple_columns(self, df: pd.DataFrame, columns: list):
        """Return multiple columns."""
        return df[columns]

    def filter_rows(self, df: pd.DataFrame, column: str, value):
        """Filter rows by condition."""
        return df[df[column] == value]

    def filter_greater_than(self, df: pd.DataFrame, column: str, value):
        """Filter rows where column > value."""
        return df[df[column] > value]

    # ---------------------------------------------------
    # SORTING
    # ---------------------------------------------------

    def sort_by_column(self, df: pd.DataFrame, column: str, ascending=True):
        """Sort dataframe by column."""
        return df.sort_values(by=column, ascending=ascending)

    # ---------------------------------------------------
    # GROUPBY OPERATIONS
    # ---------------------------------------------------

    def groupby_count(self, df: pd.DataFrame, group_col: str, count_col: str):
        """Count rows within groups."""
        return df.groupby(group_col)[count_col].count().reset_index()

    def groupby_sum(self, df: pd.DataFrame, group_col: str, sum_col: str):
        """Sum values within groups."""
        return df.groupby(group_col)[sum_col].sum().reset_index()

    def groupby_mean(self, df: pd.DataFrame, group_col: str, mean_col: str):
        """Calculate mean within groups."""
        return df.groupby(group_col)[mean_col].mean().reset_index()

    def groupby_multiple(self, df: pd.DataFrame, group_cols: list, agg_dict: dict):
        """
        Perform multiple aggregations.

        Example:
        agg_dict = {
            'salary': 'mean',
            'age': 'max'
        }
        """
        return df.groupby(group_cols).agg(agg_dict).reset_index()

    # ---------------------------------------------------
    # UNIQUE / DISTINCT
    # ---------------------------------------------------

    def get_unique(self, df: pd.DataFrame, column: str):
        """Return unique values."""
        return df[column].unique()

    def count_unique(self, df: pd.DataFrame, column: str):
        """Count unique values."""
        return df[column].nunique()

    # ---------------------------------------------------
    # DUPLICATES
    # ---------------------------------------------------

    def remove_duplicates(self, df: pd.DataFrame, column: str):
        """Remove duplicate rows."""
        return df.drop_duplicates(subset=[column])

    # ---------------------------------------------------
    # ADD / MODIFY COLUMNS
    # ---------------------------------------------------

    def create_column(self, df: pd.DataFrame, new_col: str, calculation):
        """
        Create a new column.

        Example usage:
        lambda df: df["salary"] * 1.1
        """
        df[new_col] = calculation(df)
        return df

    # ---------------------------------------------------
    # RENAME
    # ---------------------------------------------------

    def rename_column(self, df: pd.DataFrame, old: str, new: str):
        """Rename column."""
        return df.rename(columns={old: new})

    # ---------------------------------------------------
    # MERGE / JOIN
    # ---------------------------------------------------

    def merge_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame, on_col: str, how="inner"):
        """
        Merge two dataframes.

        how options:
        inner, left, right, outer
        """
        return pd.merge(df1, df2, on=on_col, how=how)

    # ---------------------------------------------------
    # PIVOT TABLES
    # ---------------------------------------------------

    def pivot_table(self, df: pd.DataFrame, index: str, values: str, agg="mean"):
        """Create pivot table."""
        return df.pivot_table(index=index, values=values, aggfunc=agg)

    # ---------------------------------------------------
    # NULL HANDLING
    # ---------------------------------------------------

    def find_nulls(self, df: pd.DataFrame):
        """Show null counts."""
        return df.isnull().sum()

    def drop_nulls(self, df: pd.DataFrame):
        """Remove rows with null values."""
        return df.dropna()

    def fill_nulls(self, df: pd.DataFrame, value):
        """Fill null values."""
        return df.fillna(value)

    # ---------------------------------------------------
    # APPLY FUNCTIONS
    # ---------------------------------------------------

    def apply_function(self, df: pd.DataFrame, column: str, func):
        """
        Apply custom function to column.

        Example:
        lambda x: x * 2
        """
        return df[column].apply(func)

    # ---------------------------------------------------
    # VALUE COUNTS
    # ---------------------------------------------------

    def value_counts(self, df: pd.DataFrame, column: str):
        """Count occurrences of values."""
        return df[column].value_counts().reset_index()

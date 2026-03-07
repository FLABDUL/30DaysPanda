import pandas as pd
import logging


class CategorizeProductsSolution:
    """
    LeetCode Pandas Problem: Group Sold Products By Date

    Goal:
        For each sell_date:
            1. Count unique products sold
            2. Return the products sorted alphabetically
            3. Join them into a comma-separated string
    """

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.logger = logging.getLogger(__name__)

    def categorize_products(self, activities: pd.DataFrame) -> pd.DataFrame:
        """
        Steps:
        1. Group rows by sell_date
        2. Aggregate the product column
        3. Count unique products (nunique)
        4. Create a sorted comma-separated product list
        """

        self.logger.info("Input DataFrame:")
        self.logger.info(f"\n{activities}")

        result = (
            activities
            .groupby('sell_date')['product']
            .agg([
                ('num_sold', 'nunique'),  # count distinct products
                ('products', lambda x: ','.join(sorted(x.unique())))
            ])
            .reset_index()
        )

        self.logger.info("\nGrouped result:")
        self.logger.info(f"\n{result}")

        return result


def example_run():
    """
    Example dataset to run the solution locally in PyCharm.
    """

    data = {
        "sell_date": [
            "2020-05-30",
            "2020-06-01",
            "2020-06-02",
            "2020-05-30",
            "2020-06-01"
        ],
        "product": [
            "Headphone",
            "Pencil",
            "Mask",
            "Basketball",
            "Book"
        ]
    }

    activities_df = pd.DataFrame(data)

    solution = CategorizeProductsSolution()
    result = solution.categorize_products(activities_df)

    print("\nReturned result:")
    print(result)


if __name__ == "__main__":
    example_run()
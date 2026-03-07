import pandas as pd
import logging


class LargestOrdersSolution:
    """
    LeetCode Pandas Problem: Largest Number of Orders

    Goal:
        Find the customer_number that appears the most times
        in the orders dataframe.

    Key Idea:
        The most frequent value in a column can be found using `.mode()`.
    """

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.logger = logging.getLogger(__name__)

    def largest_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the customer_number(s) with the most orders.

        Steps:
        1. Select the 'customer_number' column
        2. Use `.mode()` to find the most frequent value(s)
        3. Convert the result back to a DataFrame
        """

        self.logger.info("Input dataframe:")
        self.logger.info(f"\n{orders}")

        # Step 1: find the most frequent customer_number
        most_frequent = orders['customer_number'].mode()

        self.logger.info("\nMost frequent customer_number(s):")
        self.logger.info(f"\n{most_frequent}")

        # Step 2: convert Series → DataFrame
        result = most_frequent.to_frame()

        self.logger.info("\nFinal result DataFrame:")
        self.logger.info(f"\n{result}")

        return result


def example_run():
    """
    Example runner so the script can be executed directly in PyCharm.
    """

    data = {
        "order_number": [1, 2, 3, 4, 5],
        "customer_number": [1, 2, 1, 3, 1]
    }

    orders_df = pd.DataFrame(data)

    solution = LargestOrdersSolution()
    result = solution.largest_orders(orders_df)

    print("\nReturned result:")
    print(result)


if __name__ == "__main__":
    example_run()
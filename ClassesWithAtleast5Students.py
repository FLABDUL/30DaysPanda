import pandas as pd
import logging


class FindClassesSolution:
    """
    Solution class for the 'Find Classes With >=5 Students' pandas problem.

    Goal:
        Given a DataFrame with columns:
            - student
            - class

        Return a DataFrame containing classes that have at least 5 students.
    """

    def __init__(self):
        """Initialize logger."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def find_classes(self, courses: pd.DataFrame) -> pd.DataFrame:
        """
        Find classes that have at least 5 students.

        Steps:
        1. Group rows by class
        2. Count students in each class
        3. Filter classes with >=5 students
        4. Return only the 'class' column
        """

        self.logger.info("Starting computation...")
        self.logger.info(f"Input DataFrame:\n{courses}")

        # Step 1: Group by class and count students
        class_counts = (
            courses
            .groupby("class")["student"]
            .count()
            .reset_index()
        )
        # use as_index=False
        self.logger.info("Grouped and counted students per class:")
        self.logger.info(f"\n{class_counts}")

        # Step 2: Filter classes with at least 5 students
        filtered_classes = class_counts[class_counts["student"] >= 5]

        self.logger.info("Classes with >=5 students:")
        self.logger.info(f"\n{filtered_classes}")

        # Step 3: Return only the class column
        result = filtered_classes[["class"]]

        self.logger.info("Final result:")
        self.logger.info(f"\n{result}")

        return result


def example_run():
    """
    Create example input and run the solution.
    This lets you run the file directly in PyCharm.
    """

    data = {
        "student": [
            "A", "B", "C", "D", "E",
            "F", "G", "H", "I", "J",
            "K", "L"
        ],
        "class": [
            "Math", "Math", "Math", "Math", "Math",
            "Science", "Science", "Science", "Science", "Science",
            "History", "History"
        ]
    }

    courses_df = pd.DataFrame(data)

    solution = FindClassesSolution()
    result = solution.find_classes(courses_df)

    print("\nResult returned by function:")
    print(result)


if __name__ == "__main__":
    example_run()
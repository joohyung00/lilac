from tabulate import tabulate

from tabulate import tabulate

def print_table_from_list(table_data: list[list]):
    if not table_data or not isinstance(table_data[0], list):
        print("Invalid table format")
        return

    header = table_data[0]
    rows = table_data[1:]

    # Format float cells to 4 decimal places
    formatted_rows = []
    for row in rows:
        formatted_row = []
        for cell in row:
            if isinstance(cell, float):
                formatted_row.append(f"{cell:.4f}")
            else:
                formatted_row.append(cell)
        formatted_rows.append(formatted_row)

    print(tabulate(formatted_rows, headers=header, tablefmt="fancy_grid"))



if __name__ == "__main__":
    # Example usage
    table_data = [
        ["Name", "Age", "City"],
        ["Alice", 30, "New York"],
        ["Bob", 25, "Los Angeles"],
        ["Charlie", 35, "Chicago"],
    ]

    print_table_from_list(table_data)
import csv
import sys

def csv_to_markdown(csv_filepath):
    """Reads a CSV file, converts its content to a Markdown table,
    and writes the Markdown table back to the same file.

    Args:
        csv_filepath: The path to the CSV file.
    """
    with open(csv_filepath, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        filtered_header = [col for col in header if col]
        markdown_header = " | ".join([f"${col}$" for col in filtered_header])
        separator = " | ".join(["---"] * len(filtered_header))
        markdown_rows = []
        for row in reader:
            filtered_row = [cell for i, cell in enumerate(row) if header[i]]
            markdown_rows.append(" | ".join(filtered_row))

    markdown_output = f"{markdown_header}\n{separator}\n{''.join(row + '\n' for row in markdown_rows)}"

    with open(csv_filepath, 'w', newline='') as csvfile:
        csvfile.write(markdown_output)

    print(f"Successfully converted '{csv_filepath}' to Markdown and overwrote the file.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <csv_file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    csv_to_markdown(file_path)

import csv
import sys
import re

# Check if the correct number of command-line arguments are provided
if len(sys.argv) != 4:
    print("Usage: python script.py /path/to/input/csv /path/to/output/csv separator")
    sys.exit(1)

# Get the input CSV file path, output CSV file path, and separator from the command-line arguments
input_csv = sys.argv[1]
output_csv = sys.argv[2]
separator = sys.argv[3]

# Create a list to store the data
data = []

# Open and read the input CSV file
with open(input_csv, 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=separator)

    # Read the header row
    header = next(csv_reader)

    # Add a new header for the new column
    header.append('Tipo_Int')
    data.append(header)

    # Function to map Tipo values to integers (case-insensitive)
    def map_tipo_to_int(tipo):
        tipo_lower = tipo.lower()
        if re.search(r'decreto', tipo_lower):
            return 0
        elif re.search(r'portaria', tipo_lower):
            return 1
        elif re.search(r'resolução-re', tipo_lower):
            return 2
        else:
            return None

    # Iterate through the rows in the input CSV file
    for row in csv_reader:
        tipo = row[1]  # Assuming 'Tipo' is in the second column
        tipo_int = map_tipo_to_int(tipo)
        if tipo_int is not None:
            row.append(tipo_int)
        else:
            row.append('')  # Empty value if Tipo doesn't match

        data.append(row)

# Save the data with the new column to the output CSV file
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=separator)
    csv_writer.writerows(data)

print(f"Data has been written to {output_csv}")

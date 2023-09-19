import os
import sys
import xml.etree.ElementTree as ET
import csv
import re

# Check if the correct number of command-line arguments are provided
if len(sys.argv) != 3:
    print("Usage: python script.py /path/to/xml/files output.csv")
    sys.exit(1)

# Get the directory path and CSV file name from the command-line arguments
xml_directory = sys.argv[1]
csv_file = sys.argv[2]

# Open the CSV file in write mode
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header row including the new column
    csv_writer.writerow(['Identifica', 'Tipo', 'Texto'])

    # Function to extract data from an XML file and write to CSV
    def extract_data_and_write_to_csv(xml_file):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Extract Identifica and Texto fields
            identifica = root.find('.//Identifica').text
            texto = root.find('.//Texto').text

            # Use regex to remove the specified HTML tags
            texto = re.sub(r'<p class="identifica">.*?</p><p>', '', texto)

            # Remove all HTML tags using regex
            texto = re.sub(r'<[^>]*>', '', texto)

            # Use regex to extract the document type
            match = re.search(r'(PORTARIA|DESPACHO|RESOLUÇÃO-RE)', identifica, re.IGNORECASE)
            doc_type = match.group() if match else ""

            # Check if Identifica is not empty before writing the row
            if identifica:
                csv_writer.writerow([identifica, doc_type, texto])
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")

    # Iterate through XML files in the directory
    for filename in os.listdir(xml_directory):
        if filename.endswith('.xml'):
            xml_file = os.path.join(xml_directory, filename)
            extract_data_and_write_to_csv(xml_file)

print(f"Data has been written to {csv_file}")


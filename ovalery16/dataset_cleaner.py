# Function to clean the dataset and save the cleaned output
def clean_dataset(input_file_path, output_file_path):
    # Open the input file for reading
    with open(input_file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Prepare cleaned data
    cleaned_lines = []
    for line in lines:
        # Split the line by tab
        parts = line.strip().split("\t")
        
        # Check if there are at least two parts (English and Tagalog text)
        if len(parts) >= 2:
            english_text = parts[0].strip()  # First part
            tagalog_text = parts[1].strip()  # Second part
            
            # Append the cleaned line (English and Tagalog separated by a tab)
            cleaned_lines.append(f"{english_text}\t{tagalog_text}")

    # Write the cleaned data to the output file
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(cleaned_lines))
    
    print(f"Cleaned dataset has been saved to {output_file_path}")

# Specify input and output file paths
input_file_path = "tgl.txt"  # Replace with your input file path
output_file_path = "new_tgl.txt"  # Replace with your desired output file path

# Call the function to clean the dataset
clean_dataset(input_file_path, output_file_path)

# Function to clean the dataset
def clean_dataset(input_file_path, output_file_path):
    input_texts = []
    target_texts = []

    # Read the dataset
    with open(input_file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Split the line into input and target texts by tab character
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            input_text, target_text = parts[:2]  # Take only the first two parts
            input_text = input_text.strip()
            target_text = target_text.strip()

            # Remove the attribution part in target text (if present)
            if "CC-BY" in target_text:
                target_text = target_text.split("CC-BY")[0].strip()

            # Add start and end tokens to target text
            target_text = " " + target_text + " "

            input_texts.append(input_text)
            target_texts.append(target_text)

    # Save cleaned data into a new file with "English ||| Tagalog" format
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for input_text, target_text in zip(input_texts, target_texts):
            output_file.write(f"{input_text} ||| {target_text}\n")
    
    print(f"Cleaned dataset has been saved to {output_file_path}")

# Set the input and output file paths
input_file_path = "./data/tgl.txt"
output_file_path = "./data/cleaned_tgl.txt"

# Clean the dataset
clean_dataset(input_file_path, output_file_path)

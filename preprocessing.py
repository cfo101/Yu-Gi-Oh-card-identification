import os
import csv
from PIL import Image
from sklearn.model_selection import train_test_split

RAW_DIR = "yugioh_images_test"
PROCESSED_DIR = "yugioh_images_processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

IMG_SIZE = (224, 224)  # resize to 224x224
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Get all images
image_files = [f for f in os.listdir(RAW_DIR)]

# Extract card names from filenames (assumes format: ID_CardName_imgX.ext)
def extract_card_name(filename):
    # Example: "123456_BlueEyes_White_Dragon_img1.jpg"
    parts = filename.split("_")
    if len(parts) >= 2:
        return "_".join(parts[1:-1])  # everything between ID and _imgX
    return "unknown"

# Build a list of (filename, label)
data = [(f, extract_card_name(f)) for f in image_files]

# Split dataset
train_val, test = train_test_split(data, test_size=TEST_RATIO, random_state=42)
train, val = train_test_split(train_val, test_size=VAL_RATIO/(TRAIN_RATIO + VAL_RATIO), random_state=42)

splits = {"train": train, "val": val, "test": test}

# Process and save images
for split_name, split_data in splits.items():
    split_dir = os.path.join(PROCESSED_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    csv_path = os.path.join(split_dir, f"{split_name}_labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])
        
        for filename, label in split_data:
            raw_path = os.path.join(RAW_DIR, filename)
            processed_path = os.path.join(split_dir, filename)
            
            try:
                img = Image.open(raw_path).convert("RGB")  # ensure RGB
                img = img.resize(IMG_SIZE)
                img.save(processed_path)  # save resized image
                writer.writerow([filename, label])
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

print("Preprocessing complete!")
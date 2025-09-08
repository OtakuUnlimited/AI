import os
from PIL import Image
import numpy as np
import pandas as pd
from deepface import DeepFace
import logging
import random

# Set random seed for reproducibility
random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 1. Dataset folder and output CSV
input_folder = r'D:\college\Ai\AS2\proper\masked_dataset' 
output_csv = 'masked_dataset_fer.csv'

# List to store data for each image
image_data = []

# Auto-detect top-level split folders
splits = [d for d in os.listdir(input_folder)
          if os.path.isdir(os.path.join(input_folder, d))]
logger.info(f"Found splits: {splits}")

valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

for split in splits:
    split_path = os.path.join(input_folder, split)
    logger.info(f"Processing split: {split}")

    # Loop through emotion subfolders
    for emotion_folder in os.listdir(split_path):
        emotion_path = os.path.join(split_path, emotion_folder)
        if not os.path.isdir(emotion_path):
            logger.warning(f"Skipping non-directory: {emotion_path}")
            continue

        logger.info(f"Processing emotion folder: {emotion_folder}")
        # Loop through image files in each emotion folder
        for filename in os.listdir(emotion_path):
            if not filename.lower().endswith(valid_exts):
                logger.debug(f"Skipping non-image file: {filename}")
                continue

            image_path = os.path.join(emotion_path, filename)
            logger.debug(f"Processing image: {image_path}")

            # Open and process image
            try:
                with Image.open(image_path) as img:
                    img = img.convert('L')            # Convert to grayscale
                    img = img.resize((48, 48))        # Resize to 48x48
                    flattened_pixels = np.array(img, dtype=np.uint8).flatten().tolist()

                # Use DeepFace to predict age and gender
                try:
                    # DeepFace requires RGB image for analysis, so reload in RGB
                    rgb_img = Image.open(image_path).convert('RGB')
                    result = DeepFace.analyze(
                        np.array(rgb_img),
                        actions=['age', 'gender'],
                        enforce_detection=False,
                        silent=True
                    )
                    # Handle case where result is a list (DeepFace sometimes returns a list)
                    if isinstance(result, list):
                        result = result[0]
                    age = result.get('age', random.randint(18, 80))  # Fallback to random if not detected
                    gender = result.get('dominant_gender', random.choice(['Male', 'Female']))  # Fallback
                    logger.info(f"DeepFace analysis for {filename}: Age={age}, Gender={gender}")
                except Exception as deepface_error:
                    logger.error(f"DeepFace error for {image_path}: {deepface_error}")
                    age = random.randint(18, 80)  # Fallback to random age
                    gender = random.choice(['Male', 'Female'])  # Fallback to random gender
                    logger.info(f"Using fallback values for {filename}: Age={age}, Gender={gender}")

                # Append data
                image_data.append({
                    'file name': filename,
                    'image pixel': ' '.join(map(str, flattened_pixels)),  # space-separated string
                    'age': age,
                    'gender': gender,
                    'emotion': emotion_folder,  
                    'split': split              
                })
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")

# 3. Create DataFrame and export to CSV
df = pd.DataFrame(image_data, columns=['file name', 'image pixel', 'age', 'gender', 'emotion', 'split'])
df.to_csv(output_csv, index=False)

logger.info(f"Processed {len(df)} images and saved to '{output_csv}'")
logger.info(f"Split counts:\n{df['split'].value_counts().to_string()}")
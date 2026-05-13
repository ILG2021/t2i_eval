import pandas as pd
import os

def create_sample_excel():
    data = {
        "英文提示词": [
            "A beautiful sunset over the mountains, 4k, highly detailed",
            "A futuristic city with flying cars, cyberpunk style, neon lights",
            "A cute cat wearing a spacesuit on Mars, cinematic lighting"
        ],
        "图片名": [
            "mountain_sunset",
            "cyberpunk_city",
            "space_cat"
        ]
    }
    
    df = pd.DataFrame(data)
    file_path = "data.xlsx"
    df.to_excel(file_path, index=False)
    print(f"Sample Excel file created successfully at: {os.path.abspath(file_path)}")

if __name__ == "__main__":
    create_sample_excel()

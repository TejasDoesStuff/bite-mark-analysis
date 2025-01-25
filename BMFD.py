"""
Bite Mark Feature Detector (BMFD)
Image analyzer that looks through each image, finds the edges of the bite marks, and determines depth, width, and bite force of the front two teeth.

Inputs:
Image of bite mark

Outputs:
Alterior Teeth Width (mm),
Alterior Teeth Depth (mm),
Bite Force (N)
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import zipfile
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

#extract images from zip file
zip_path = "bite_mark_images.zip"
image_dir = "extracted_images/bite_mark_images"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("extracted_images")

csv_path = "bite_mark_data.csv"
bite_data = pd.read_csv(csv_path)
bite_data = bite_data.dropna(subset=['Number'])
bite_data = bite_data.loc[:, ~bite_data.columns.str.contains('^Unnamed')]
bite_data['Number'] = bite_data['Number'].astype(int)

def augment_image(image):
    augmented_images = []
    for _ in range(10):  # 10 augents per image = 300 images
        augmented_image = image.copy()

        # reflect
        if np.random.rand() > 0.5:
            augmented_image = cv2.flip(augmented_image, 1)

        # angle
        angle = np.random.randint(-15, 15)
        h, w = augmented_image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        augmented_image = cv2.warpAffine(augmented_image, M, (w, h))

        # brightness
        brightness_factor = np.random.uniform(0.5, 1.5)
        augmented_image = cv2.convertScaleAbs(augmented_image, alpha=brightness_factor)

        # noise
        noise = np.random.rand(*augmented_image.shape)
        augmented_image[noise < 0.02] = 0
        augmented_image[noise > 0.98] = 255

        # cropping
        if np.random.rand() > 0.5:
            x_start = np.random.randint(0, w//4)
            y_start = np.random.randint(0, h//4)
            augmented_image = augmented_image[y_start:y_start + h//2, x_start:x_start + w//2]

        augmented_images.append(augmented_image)

    return augmented_images

# processes each image
def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tooth_measurements = []

    augmented_images = augment_image(image)

    for aug_img in augmented_images:
        blurred_image = cv2.GaussianBlur(aug_img, (5, 5), 0)
        edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 5 and h > 5:
                tooth_measurements.append({'width': w, 'depth': h})

    if len(tooth_measurements) >= 2:
        tooth_measurements.sort(key=lambda x: x['width'])
        return tooth_measurements[:2]

# predicts the depth, width, bite force for the image
predictions = []
for index, row in bite_data.iterrows():
    number = row['Number']
    image_name = f"{int(number)}.jpg"
    image_path = os.path.join(image_dir, image_name)

    if os.path.exists(image_path):
        measurements = process_image(image_path)

        if len(measurements) >= 2:
            left_tooth = measurements[0]
            right_tooth = measurements[1]
            predictions.append({
                'Number': number,
                'Predicted Left Width': left_tooth['width'],
                'Predicted Right Width': right_tooth['width'],
                'Predicted Left Depth': left_tooth['depth'],
                'Predicted Right Depth': right_tooth['depth']
            })

X = bite_data[['Left Tooth Width (mm)', 'Right Tooth Width (mm)', 'Left Tooth Depth (mm)', 'Right Tooth Depth (mm)']]
y = bite_data['Bite Force (N)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# splits data into testing (20%) and training (80%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values.ravel(), test_size=0.2, random_state=42)

reg_model = LinearRegression().fit(X_train, y_train)

for prediction in predictions:
    features = [
        prediction['Predicted Left Width'],
        prediction['Predicted Right Width'],
        prediction['Predicted Left Depth'],
        prediction['Predicted Right Depth']
    ]

    scaled_features = scaler.transform([features])
    prediction['Predicted Bite Force'] = reg_model.predict(scaled_features)[0]

predictions_df = pd.DataFrame(predictions)
result_df = pd.merge(bite_data[['Number', 'Left Tooth Width (mm)', 'Right Tooth Width (mm)',
                                  'Left Tooth Depth (mm)', 'Right Tooth Depth (mm)',
                                  'Bite Force (N)']], predictions_df,
                     on="Number", how="left")

output_path = "predicted_bite_force.csv"
result_df.to_csv(output_path, index=False)

test_results = pd.DataFrame(X_test)
test_results['Actual Bite Force'] = y_test
test_results['Predicted Bite Force'] = reg_model.predict(X_test)

for i in range(len(test_results)):
    number_indexed_row = bite_data.iloc[i]
    test_results.at[i, 'Image Number'] = number_indexed_row['Number']
    test_results.at[i, 'Actual Left Width'] = number_indexed_row['Left Tooth Width (mm)']
    test_results.at[i, 'Actual Right Width'] = number_indexed_row['Right Tooth Width (mm)']
    test_results.at[i, 'Actual Left Depth'] = number_indexed_row['Left Tooth Depth (mm)']
    test_results.at[i, 'Actual Right Depth'] = number_indexed_row['Right Tooth Depth (mm)']

    pred_row_index = predictions_df[predictions_df['Number'] == number_indexed_row['Number']]

    if not pred_row_index.empty:
        test_results.at[i, 'Predicted Left Width'] = pred_row_index.iloc[0]['Predicted Left Width']
        test_results.at[i, 'Predicted Right Width'] = pred_row_index.iloc[0]['Predicted Right Width']
        test_results.at[i, 'Predicted Left Depth'] = pred_row_index.iloc[0]['Predicted Left Depth']
        test_results.at[i, 'Predicted Right Depth'] = pred_row_index.iloc[0]['Predicted Right Depth']

reshaped_results_list = []

for i in range(len(test_results)):
    reshaped_results_list.append({
        'Image Number': test_results.at[i, 'Image Number'],
        'Feature': 'Actual Bite Force',
        'Value': test_results.at[i, 'Actual Bite Force'],
        'Predicted Value': test_results.at[i, 'Predicted Bite Force']
    })

    reshaped_results_list.append({
        'Image Number': test_results.at[i, 'Image Number'],
        'Feature': 'Actual Left Width',
        'Value': test_results.at[i, 'Actual Left Width'],
        'Predicted Value': test_results.at[i, 'Predicted Left Width']
    })

    reshaped_results_list.append({
        'Image Number': test_results.at[i, 'Image Number'],
        'Feature': 'Actual Right Width',
        'Value': test_results.at[i, 'Actual Right Width'],
        'Predicted Value': test_results.at[i, 'Predicted Right Width']
    })

    reshaped_results_list.append({
        'Image Number': test_results.at[i, 'Image Number'],
        'Feature': 'Actual Left Depth',
        'Value': test_results.at[i, 'Actual Left Depth'],
        'Predicted Value': test_results.at[i, 'Predicted Left Depth']
    })

    reshaped_results_list.append({
        'Image Number': test_results.at[i, 'Image Number'],
        'Feature': 'Actual Right Depth',
        'Value': test_results.at[i, 'Actual Right Depth'],
        'Predicted Value': test_results.at[i, 'Predicted Right Depth']
    })

    reshaped_results_list.append({})

reshaped_df = pd.DataFrame(reshaped_results_list)

# outputs the results to a new file as well as the console
test_output_path = "predicted_test_bite_force.csv"
reshaped_df.to_csv(test_output_path, index=False)

r_squared_values = {
    "Bite Force": r2_score(test_results['Actual Bite Force'], test_results['Predicted Bite Force']),
}

r_squared_values["Left Width"] = r2_score(test_results['Actual Left Width'], test_results['Predicted Left Width'])
r_squared_values["Right Width"] = r2_score(test_results['Actual Right Width'], test_results['Predicted Right Width'])
r_squared_values["Left Depth"] = r2_score(test_results['Actual Left Depth'], test_results['Predicted Left Depth'])
r_squared_values["Right Depth"] = r2_score(test_results['Actual Right Depth'], test_results['Predicted Right Depth'])

print("R-squared Values:")
for feature in r_squared_values:
    print(f"{feature}: {r_squared_values[feature]:.4f}")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

print("All Test Results:")
print(reshaped_df)

# graphing
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.scatter(test_results['Actual Bite Force'], test_results['Predicted Bite Force'], color='blue')
plt.plot([min(test_results['Actual Bite Force']), max(test_results['Actual Bite Force'])],
         [min(test_results['Actual Bite Force']), max(test_results['Actual Bite Force'])], color='red', linestyle='--')
plt.title('Bite Force: Actual vs Predicted')
plt.xlabel('Actual Bite Force')
plt.ylabel('Predicted Bite Force')

plt.subplot(1, 3, 2)
plt.scatter(test_results['Actual Left Width'], test_results['Predicted Left Width'], color='green')
plt.plot([min(test_results['Actual Left Width']), max(test_results['Actual Left Width'])],
         [min(test_results['Actual Left Width']), max(test_results['Actual Left Width'])], color='red', linestyle='--')
plt.title('Left Tooth Width: Actual vs Predicted')
plt.xlabel('Actual Left Width')
plt.ylabel('Predicted Left Width')

plt.subplot(1, 3, 3)
plt.scatter(test_results['Actual Right Width'], test_results['Predicted Right Width'], color='orange')
plt.plot([min(test_results['Actual Right Width']), max(test_results['Actual Right Width'])],
         [min(test_results['Actual Right Width']), max(test_results['Actual Right Width'])], color='red', linestyle='--')
plt.title('Right Tooth Width: Actual vs Predicted')
plt.xlabel('Actual Right Width')
plt.ylabel('Predicted Right Width')

plt.tight_layout()
plt.show()

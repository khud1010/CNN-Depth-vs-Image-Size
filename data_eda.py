#%%
import pandas as pd
import numpy as np
from PIL import Image
import os, shutil
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

images_df = pd.read_csv('./CXR8/Data_Entry_2017_v2020.csv')
images_df['Finding Labels'] = images_df['Finding Labels'].str.split('|').str[0]

category_percentages = images_df['Finding Labels'].value_counts(normalize=True) * 100


labels = category_percentages.index
sizes = category_percentages.values

# Create the pie chart
plt.figure(figsize=(10, 8))  # Adjust figure size to give the plot more room
wedges, texts = plt.pie(sizes, startangle=140, wedgeprops=dict(width=0.5), pctdistance=0.85)

# Draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Adding legend with percentages
formatted_labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(labels, sizes)]
plt.legend(wedges, formatted_labels, title="Labels", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

plt.title('Pie Chart of Finding Labels')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

# %%
test_list = []
train_list = []

# Read training list
with open('./CXR8/train_val_list.txt', 'r') as f:
    full_train_list = [line.strip() for line in f]

# Read test list
with open('./CXR8/test_list.txt', 'r') as f:
    full_test_list = [line.strip() for line in f]

print('LENGTH OF DATA ')
print(f'Total Images : {len(images_df)}')
print(f'Train Images : {len(full_train_list)}')
print(f'Test Images : {len(full_test_list)}')


#%%
for idx, row in images_df.iterrows():
    image_name = row['Image Index'] 
    if image_name in full_train_list:
        train_list.append(row)
    elif image_name in full_test_list:
        test_list.append(row)

train_df = pd.DataFrame(train_list)
test_df = pd.DataFrame(test_list)

#%%
root_directory = 'D:/CXR_DATA/images/'

#%%
train_directory = 'D:/CXR_DATA/full_dataset/train_images/'
# Create directories based on unique labels
unique_labels = train_df['Finding Labels'].unique()
for label in unique_labels:
    label_directory = os.path.join(train_directory, label)
    os.makedirs(label_directory, exist_ok=True)


#%%
# Move images to respective directories
for index, row in train_df.iterrows():
    image_path = root_directory + row['Image Index']  # Assuming images are in the same directory as the script
    label = row['Finding Labels']
    destination_directory = os.path.join(train_directory, label)
    try:
        shutil.move(image_path, destination_directory)
    except FileNotFoundError as e:
        print(e)

print("Images moved to respective directories.")

#%%
test_directory = 'D:/CXR_DATA/full_dataset/test_images/'
#%%
# Create directories based on unique labels
unique_labels = test_df['Finding Labels'].unique()
for label in unique_labels:
    label_directory = os.path.join(test_directory, label)
    os.makedirs(label_directory, exist_ok=True)


#%%
# Move images to respective directories
for index, row in test_df.iterrows():
    image_path = root_directory + row['Image Index']  # Assuming images are in the same directory as the script
    label = row['Finding Labels']
    destination_directory = os.path.join(test_directory, label)
    try:
        shutil.move(image_path, destination_directory)
    except FileNotFoundError as e:
        print(e)

print("Images moved to respective directories.")





# %%
import os
import shutil

def consolidate_folders(root_dir, new_root):
    # Create new root directory if it does not exist
    if not os.path.exists(new_root):
        os.makedirs(new_root)

    # The subdirectories "train" and "test" under the main directory
    sub_dirs = ['train_images', 'test_images']

    for sub_dir in sub_dirs:
        old_path = os.path.join(root_dir, sub_dir)
        new_path = os.path.join(new_root, sub_dir)

        # Ensure new subdirectories exist
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        # Directory for 'Unhealthy' in the new structure
        unhealthy_path = os.path.join(new_path, 'Unhealthy')
        if not os.path.exists(unhealthy_path):
            os.makedirs(unhealthy_path)

        # Directory for 'No Findings' in the new structure
        no_findings_new_path = os.path.join(new_path, 'No Findings')
        if not os.path.exists(no_findings_new_path):
            os.makedirs(no_findings_new_path)

        # List all folders in the old directory
        folders = [f for f in os.listdir(old_path) if os.path.isdir(os.path.join(old_path, f))]

        # Loop through all folders and copy files to the new structure
        for folder in folders:
            folder_path = os.path.join(old_path, folder)
            
            # Determine if we copy to 'Unhealthy' or 'No Findings'
            if folder == 'No Finding':
                target_path = no_findings_new_path
            else:
                target_path = unhealthy_path
            
            # Copy each file in the current folder to the appropriate new folder
            for filename in os.listdir(folder_path):
                src_file = os.path.join(folder_path, filename)
                dest_file = os.path.join(target_path, filename)
                
                # Check for file existence to avoid overwriting
                if os.path.exists(dest_file):
                    base, extension = os.path.splitext(filename)
                    count = 1
                    new_filename = f"{base}_{count}{extension}"
                    dest_file = os.path.join(target_path, new_filename)
                    while os.path.exists(dest_file):
                        count += 1
                        new_filename = f"{base}_{count}{extension}"
                        dest_file = os.path.join(target_path, new_filename)

                shutil.copy(src_file, dest_file)


# Set the root directory where 'train' and 'test' folders are located
root_directory = 'D:/CXR_DATA/full_dataset'  # Update this to your root directory path
new_directory = 'D:/CXR_DATA/binary_data'  # Update this to your root directory path
consolidate_folders(root_directory, new_directory)
print("Consolidation complete.")

# %%

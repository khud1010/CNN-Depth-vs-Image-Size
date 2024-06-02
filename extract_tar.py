#%%
import tarfile
import os

def extract_tar_gz(source_folder, target_folder):
    # Check if the target folder exists, if not create it
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # List all files in the source folder
    files = [f for f in os.listdir(source_folder) if f.endswith('.tar.gz')]
    
    # Loop through each tar.gz file and extract it
    for file in files:
        # Path to your tar.gz file
        file_path = os.path.join(source_folder, file)
        
        # Open the tar.gz file
        with tarfile.open(file_path, 'r:gz') as tar:
            # Extract all contents into the target folder
            tar.extractall(path=target_folder)
            print(f"Extracted {file} into {target_folder}")

# Define the source and target directories
target_folder = r'D:\CXR_DATA'
script_dir = os.path.dirname(os.path.abspath(__file__))
source_folder = os.path.join(script_dir, '..', 'DL Project', 'CXR8','images')
source_folder = os.path.normpath(source_folder)  
#%%
# Call the function to start extraction
extract_tar_gz(source_folder, target_folder)


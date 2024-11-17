import os
import re

# Folder containing your files
folder_path = "/path/to/your/folder"

# Regular expression to capture season and episode info
pattern = re.compile(r'[Ss](\d+)[Ee](\d+)')


# Function to rename files
def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Extract file extension
        ext = os.path.splitext(filename)[1]

        # Try to find season and episode info in the filename
        match = pattern.search(filename)
        if match:
            season = int(match.group(1))
            episode = int(match.group(2))

            # New filename based on season and episode
            if season:
                new_name = f"s{season:02}e{episode:02}{ext.lower()}"
            else:
                new_name = f"e{episode:02}{ext.lower()}"

            new_file_path = os.path.join(folder_path, new_name)

            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"Renamed '{filename}' to '{new_name}'")
        else:
            print(f"Skipped '{filename}' (no season/episode info found)")


# Call the function
rename_files(folder_path)

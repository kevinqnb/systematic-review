import os

def get_filenames_in_directory(directory_path):
    """
    Returns a list of all filenames in the specified directory.
    
    Args:
        directory_path (str): The path to the directory.
    
    Returns:
        list: A list of filenames in the directory.
    """
    try:
        filenames = [
            f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))
        ]
        return filenames
    except FileNotFoundError:
        return f"Error: Directory not found: {directory_path}"
    except NotADirectoryError:
         return f"Error: Not a directory: {directory_path}"
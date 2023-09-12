import os
import re

from visitors.remove_visitor import Remove_visitor
from functions.process_content_recursively import process_content_recursively

def remove_dir_with_content(root_path: str):
    process_content_recursively(root_path, Remove_visitor())
    os.rmdir(root_path)
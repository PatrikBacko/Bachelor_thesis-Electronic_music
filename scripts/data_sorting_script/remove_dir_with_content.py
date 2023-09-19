import os
import re

from process_dir_content import process_dir_content
from remove_visitor import Remove_visitor

def remove_dir_with_content(root_path: str):
    process_dir_content(root_path, Remove_visitor())
    os.rmdir(root_path)
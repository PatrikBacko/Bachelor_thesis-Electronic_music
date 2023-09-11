import os
import re

from visitors import IVisitor, RemoveVisitor

def remove_dir_with_content(root_path: str):
    process_content_recursively(root_path, RemoveVisitor())
    os.rmdir(root_path)

def process_content_recursively(root_path: str, visitor: IVisitor):
    for name in os.listdir(root_path):
        path = os.path.join(root_path, name)
        if os.path.isdir(path):
            if visitor.dir_predicate(path):
                process_content_recursively(path, visitor)
            visitor.dir_action(path)
        else:
            visitor.file_action(path)
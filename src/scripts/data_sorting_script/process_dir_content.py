import os
import re

from visitor import Visitor

def process_dir_recursion(root_path: str, visitor: Visitor):
    for name in os.listdir(root_path):
        path = os.path.join(root_path, name)
        if os.path.isdir(path):
            if visitor.dir_predicate(path):
                process_dir_recursion(path, visitor)
            visitor.dir_action(path)

        elif visitor.file_predicate(path):
            os.chmod(path, 0o777)
            visitor.file_action(path)


def process_dir_content(root_path: str, visitor: Visitor):
    if os.path.isdir(root_path):
        process_dir_recursion(root_path, visitor)
    else:
        raise Exception("Path is not a directory")
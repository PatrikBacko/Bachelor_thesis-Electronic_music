import os
import re

from visitors.IVisitor import IVisitor
from functions.remove_dir_with_content import remove_dir_with_content

class Remove_MACOSX_dirs_visitor(IVisitor):
    def file_action(self, path: str):
        pass
    def dir_action(self, path):
        pass
    def dir_predicate(self, path: str) -> bool:
        if os.path.basename(path) == "__MACOSX":
            remove_dir_with_content(path)
            return False
        return True
import os
import re

from process_files import remove_dir_with_content

class IVisitor:
    def file_action(self, path: str):
        pass
    def dir_action(self, path):
        pass
    def dir_predicate(self, path: str) -> bool:
        return True

class RemoveVisitor(IVisitor):
    def file_action(self, path: str):
        os.remove(path)
    def dir_action(self, path):
        os.rmdir(path)
    def dir_predicate(self, path: str) -> bool:
        return True

class Visitor1(IVisitor):
    def file_action(self, path: str):
        pass
    def dir_action(self, path):
        pass
    def dir_predicate(self, path: str) -> bool:
        if os.path.basename(path) == "__MACOSX":
            remove_dir_with_content(path)
            return False
        return True
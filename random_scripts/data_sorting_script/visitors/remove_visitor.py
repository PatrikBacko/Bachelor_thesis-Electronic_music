import os

from visitors.IVisitor import IVisitor

class Remove_visitor(IVisitor):
    def file_action(self, path: str):
        os.remove(path)
    def dir_action(self, path):
        os.rmdir(path)
    def dir_predicate(self, path: str) -> bool:
        return True
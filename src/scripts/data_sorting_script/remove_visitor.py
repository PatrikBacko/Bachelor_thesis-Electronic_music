import os

from visitor import Visitor

class Remove_visitor(Visitor):
    def file_action(self, path: str):
        os.remove(path)
    def dir_action(self, path):
        os.rmdir(path)
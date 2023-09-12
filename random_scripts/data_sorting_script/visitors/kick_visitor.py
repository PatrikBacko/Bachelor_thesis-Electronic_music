import re
import os

from visitors.IVisitor import IVisitor 

class Kick_visitor(IVisitor):
    def __init__(self):
        self.counter = 0
    def file_action(self, path: str):
        if re.search(r"kick", path, re.IGNORECASE):
            self.counter += 1
    def dir_action(self, path):
        pass
    def dir_predicate(self, path: str) -> bool:
        return True
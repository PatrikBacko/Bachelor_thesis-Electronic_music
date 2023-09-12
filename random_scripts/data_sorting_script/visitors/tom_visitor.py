import os
import re

from visitors.IVisitor import IVisitor

class Tom_visitor(IVisitor):
    def __init__(self):
        self.counter = 0
    def file_action(self, path: str):
        if re.search(r"tom", path, re.IGNORECASE):
            self.counter += 1
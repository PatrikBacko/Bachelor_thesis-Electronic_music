import os
import re

from visitors.IVisitor import IVisitor

class Snare_visitor(IVisitor):
    def __init__(self):
        self.counter = 0
    def file_action(self, path: str):
        if re.search(r"snare", path, re.IGNORECASE):
            self.counter += 1
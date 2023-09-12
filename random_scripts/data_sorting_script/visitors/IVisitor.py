import os
import re

class IVisitor:
    def file_action(self, path: str):
        pass
    def dir_action(self, path):
        pass
    def dir_predicate(self, path: str) -> bool:
        return True

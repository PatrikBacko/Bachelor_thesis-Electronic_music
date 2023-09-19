import os
import re

from visitor import Visitor
from remove_dir_with_content import remove_dir_with_content


class Remove_MACOSX_dirs_visitor(Visitor):
    def dir_predicate(self, path: str) -> bool:
        if os.path.basename(path) == "__MACOSX":
            remove_dir_with_content(path)
            return False
        return True
        

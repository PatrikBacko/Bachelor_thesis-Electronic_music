import os
import sys
import re

from visitors.IVisitor import IVisitor
from visitors.tom_visitor import Tom_visitor
from visitors.snare_visitor import Snare_visitor
from visitors.kick_visitor import Kick_visitor
from visitors.remove_visitor import Remove_visitor
from visitors.remove_MACOSX_dirs_visitor import Remove_MACOSX_dirs_visitor

from functions.process_content_recursively import process_content_recursively
from functions.remove_dir_with_content import remove_dir_with_content

def main():
    path = r"C:\Users\llama\Desktop\programming shit\Bakalarka\Bakalaris-data\uzipped-samples"

    visitor = Tom_visitor()
    process_content_recursively(path, visitor)
    print(f"Toms Count: {visitor.counter}")

    visitor = Snare_visitor()
    process_content_recursively(path, visitor)
    print(f"Snares Count: {visitor.counter}")

    visitor = Kick_visitor()
    process_content_recursively(path, visitor)
    print(f"Kick Count: {visitor.counter}")

    # visitor = Remove_MACOSX_dirs()
    # process_content_recursively(path, visitor)

    # visitor = IVisitor()
    # process_content_recursively(path, visitor)


if __name__ == "__main__":
    main()

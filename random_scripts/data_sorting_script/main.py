import os
import sys
import re

from visitors import Visitor1
from process_files import process_content_recursively
    
def main():
    path = r"C:\Users\llama\Downloads\xdd"
    visitor = Visitor1()
    process_content_recursively(path, visitor)

if __name__ == "__main__":
    main()


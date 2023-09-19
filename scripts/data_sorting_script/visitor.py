class Visitor:
    def file_predicate(self, path: str) -> bool:
        return True
    def file_action(self, path: str):
        pass
    def dir_predicate(self, path: str) -> bool:
        return True
    def dir_action(self, path):
        pass

class Delegate_visitor(Visitor):
    def __init__(self, 
                 file_predicate: callable, 
                 file_action: callable, 
                 dir_predicate: callable,
                 dir_action: callable):
        
        self.file_action = file_action
        self.file_predicate = file_predicate
        self.dir_action = dir_action
        self.dir_predicate = dir_predicate

    def file_action(self, path: str):
        self.file_action(path)
    def file_predicate(self, path: str) -> bool:
        return self.file_predicate(path)
    def dir_action(self, path):
        self.dir_action(path)
    def dir_predicate(self, path: str) -> bool:
        return self.dir_predicate(path)
    

class File_Delegate_visitor(Visitor):
    def __init__(self, predicate: callable, action: callable):
        self.predicate = predicate
        self.action = action
    def file_action(self, path: str):
        self.action(path)
    def file_predicate(self, path: str) -> bool:
        return self.predicate(path)
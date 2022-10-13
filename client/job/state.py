from enum import Enum

class JobState(Enum): 
    EVENT = 1
    PENDING = 2
    RUNNING = 3 
    RUNNABLE = 4
    END = 5


class DDLState(Enum): 
    BEST = "best"
    STRICT ="strict"
    SOFT ="soft"

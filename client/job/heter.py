from .base import BaseJob


class HeterogeneousJob(BaseJob): 
    __alias__ = 'heterogeneous' 
    def __init__(self, df): 
        super(HeterogeneousJob, self).__init__(name=df.name, application=df.application, submission_time=df.submission_time)

import warnings
from abc import ABCMeta, abstractmethod


class BaseScheduler(metaclass=ABCMeta):
    def __init__(self, job_manager, cluster_manager, user_manager, placement, name, logger, **kwargs):
        self.job_manager = job_manager
        self.cluster_manager = cluster_manager
        self.user_manager = user_manager
        self.placement = placement
        self.name = name
        self.logger = logger
        self.full_resource_list = list()
        self.free_resource_list = list()
        self.pending_job_num_list = list()
        self.running_job_num_list = list()
        self.submit_job_num_list = list()
        self.save_dir = kwargs.get('save_dir', 'result/')



    @abstractmethod
    def check_resource(self, **kwargs):
        raise NotImplementedError
    

    @abstractmethod
    def place_jobs(self, **kwargs):
        raise NotImplementedError
    

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError
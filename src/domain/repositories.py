from abc import ABC, abstractmethod

class JobRepository(ABC):
    @abstractmethod
    def create(self, dataset_id: int): ...

    @abstractmethod
    def get(self, job_id: int): ...

    @abstractmethod
    def update_status(self, job_id: int, status: str, error: str = None): ...

class DatasetRepository(ABC):
    @abstractmethod
    def create(self, filename: str, file_path: str): ...

    @abstractmethod
    def get(self, dataset_id: int): ...
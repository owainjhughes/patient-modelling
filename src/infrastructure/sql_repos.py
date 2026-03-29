from ..domain.repositories import JobRepository, DatasetRepository
from ..domain.models import Job, Dataset
from .database import SessionLocal
from datetime import datetime

class SQLJobRepository(JobRepository):
    def create(self, dataset_id: int) -> Job:
        with SessionLocal() as db:
            job = Job(dataset_id=dataset_id)
            db.add(job)
            db.commit()
            db.refresh(job)
            return job

    def get(self, job_id: int) -> Job | None:
        with SessionLocal() as db:
            return db.query(Job).filter(Job.id == job_id).first()

    def update_status(self, job_id: int, status: str, error: str = None):
        with SessionLocal() as db:
            db.query(Job).filter(Job.id == job_id).update({
                "status": status,
                "error": error,
                "updated_at": datetime.utcnow()
            })
            db.commit()

class SQLDatasetRepository(DatasetRepository):
    def create(self, filename: str, file_path: str) -> Dataset:
        with SessionLocal() as db:
            dataset = Dataset(filename=filename, file_path=file_path)
            db.add(dataset)
            db.commit()
            db.refresh(dataset)
            return dataset

    def get(self, dataset_id: int) -> Dataset | None:
        with SessionLocal() as db:
            return db.query(Dataset).filter(Dataset.id == dataset_id).first()
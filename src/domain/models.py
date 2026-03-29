# src/domain/models.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float
from datetime import datetime
from ..infrastructure.database import Base

class Dataset(Base):
    __tablename__ = "datasets"
    id         = Column(Integer, primary_key=True)
    filename   = Column(String, nullable=False)
    file_path  = Column(String, nullable=False)  # MinIO object key
    created_at = Column(DateTime, default=datetime.utcnow)

class Job(Base):
    __tablename__ = "jobs"
    id         = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    status     = Column(String, default="pending")  # pending | processing | complete | failed
    error      = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RFResult(Base):
    __tablename__ = "rf_results"
    id          = Column(Integer, primary_key=True)
    job_id      = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    result_path = Column(String, nullable=False)  # MinIO object key
    accuracy    = Column(Float, nullable=True)

class ClusterResult(Base):
    __tablename__ = "cluster_results"
    id          = Column(Integer, primary_key=True)
    job_id      = Column(Integer, ForeignKey("jobs.id"), nullable=False)
    result_path = Column(String, nullable=False)  # MinIO object key
    n_clusters  = Column(Integer, nullable=True)
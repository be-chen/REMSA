from pydantic import BaseModel
from typing import List, Optional, Union

class MinPerformance(BaseModel):
    metric: List[str]
    value: List[float]

class QuerySchema(BaseModel):
    task: str
    modality: str
    application: Optional[str]
    sensor: Optional[Union[str, List[str]]]
    spatial_resolution: Optional[Union[str, float]]
    temporal_resolution: Optional[Union[str, float]]
    bands: Optional[List[str]]
    deployment_device: Optional[str]
    min_performance: Optional[MinPerformance]
    region: Optional[Union[str, List[str]]]
    domain_keywords: Optional[List[str]]
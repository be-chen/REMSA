from typing import List, Optional, Dict, Union, Literal
from pydantic import BaseModel, Field, HttpUrl
from datetime import date
from langchain_core.output_parsers import JsonOutputParser


class ExtractedString(BaseModel):
    value: Optional[str] = Field(description="Extracted text.")
    confidence: float = Field(description="Confidence score 0-1.")

class ExtractedStringList(BaseModel):
    value: Optional[List[ExtractedString]] = Field(description="List of extracted strings.")
    #confidence: float = Field(description="Confidence score 0-1.")

class ExtractedDict(BaseModel):
    value: Optional[Dict[str, str]] = Field(description="Extracted dictionary.")
    confidence: float = Field(description="Confidence score 0-1.")

class ExtractedNumber(BaseModel):
    value: Optional[Union[int, float]] = Field(description="Extracted numeric value.")
    confidence: float = Field(description="Confidence score 0-1.")

class ExtractedNumberList(BaseModel):
    value: Optional[List[ExtractedNumber]] = Field(description="List of extracted numeric values.")

class ExtractedAlignment(BaseModel):
    value: Optional[Literal["full", "partial", "none"]] = Field(
        description="Extracted boolean value as string ('true' or 'false')."
    )
    confidence: float = Field(description="Confidence score 0-1.")

# ==================================================================== #
#                                NER TASK                              #
# ==================================================================== #
class Entity(BaseModel):
    name : str = Field(description="The specific name of the entity. ")
    type : str = Field(description="The type or category that the entity belongs to.")
class EntityList(BaseModel):
    entity_list : List[Entity] = Field(description="Named entities appearing in the text.")

# ==================================================================== #
#                               RE TASK                                #
# ==================================================================== #
class Relation(BaseModel):
    head : str = Field(description="The starting entity in the relationship.")
    tail : str = Field(description="The ending entity in the relationship.")
    relation : str = Field(description="The predicate that defines the relationship between the two entities.")

class RelationList(BaseModel):
    relation_list : List[Relation] = Field(description="The collection of relationships between various entities.")

# ==================================================================== #
#                               EE TASK                                #
# ==================================================================== #
class Event(BaseModel):
    event_type : str = Field(description="The type of the event.")
    event_trigger : str = Field(description="A specific word or phrase that indicates the occurrence of the event.")
    event_argument : dict = Field(description="The arguments or participants involved in the event.")

class EventList(BaseModel):
    event_list : List[Event] = Field(description="The events presented in the text.")

# ==================================================================== #
#                            Triple TASK                               #
# ==================================================================== #
class Triple(BaseModel):
    head: str = Field(description="The subject or head of the triple.")
    head_type: str = Field(description="The type of the subject entity.")
    relation: str = Field(description="The predicate or relation between the entities.")
    relation_type: str = Field(description="The type of the relation.")
    tail: str = Field(description="The object or tail of the triple.")
    tail_type: str = Field(description="The type of the object entity.")
class TripleList(BaseModel):
    triple_list: List[Triple] = Field(description="The collection of triples and their types presented in the text.")

# ==================================================================== #
#                          TEXT DESCRIPTION                            #
# ==================================================================== #
class TextDescription(BaseModel):
    field: str = Field(description="The field of the given text, such as 'Science', 'Literature', 'Business', 'Medicine', 'Entertainment', etc.")
    genre: str = Field(description="The genre of the given text, such as 'Article', 'Novel', 'Dialog', 'Blog', 'Manual','Expository', 'News Report', 'Research Paper', etc.")

# ==================================================================== #
#                        USER DEFINED SCHEMA                           #
# ==================================================================== #

# --------------------------- Research Paper ----------------------- #
class MetaData(BaseModel):
    title : str = Field(description="The title of the article")
    authors : List[str] = Field(description="The list of the article's authors")
    abstract: str = Field(description="The article's abstract")
    key_words: List[str] = Field(description="The key words associated with the article")

class Baseline(BaseModel):
    method_name : str = Field(description="The name of the baseline method")
    proposed_solution : str = Field(description="the proposed solution in details")
    performance_metrics : str = Field(description="The performance metrics of the method and comparative analysis")

class ExtractionTarget(BaseModel):

    key_contributions: List[str] = Field(description="The key contributions of the article")
    limitation_of_sota : str=Field(description="the summary limitation of the existing work")
    proposed_solution : str = Field(description="the proposed solution in details")
    baselines : List[Baseline] = Field(description="The list of baseline methods and their details")
    performance_metrics : str = Field(description="The performance metrics of the method and comparative analysis")
    paper_limitations : str=Field(description="The limitations of the proposed solution of the paper")

# --------------------------- News ----------------------- #
class Person(BaseModel):
    name: str = Field(description="The name of the person")
    identity: Optional[str] = Field(description="The occupation, status or characteristics of the person.")
    role: Optional[str] = Field(description="The role or function the person plays in an event.")

class Event(BaseModel):
    name: str = Field(description="Name of the event")
    time: Optional[str] = Field(description="Time when the event took place")
    people_involved: Optional[List[Person]] = Field(description="People involved in the event")
    cause: Optional[str] = Field(default=None, description="Reason for the event, if applicable")
    process: Optional[str] = Field(description="Details of the event process")
    result: Optional[str] = Field(default=None, description="Result or outcome of the event")

# class NewsReport_conf(BaseModel):
#     title: ExtractedString = Field(description="The title or headline of the news report.")
#     summary: ExtractedString = Field(description="A brief summary of the news report.")
#     publication_date: Optional[ExtractedString] = Field(description="The publication date of the report.")
#     keywords: Optional[ExtractedStringList] = Field(description="List of keywords or topics covered in the news report.")
#     #events: List['Event'] = Field(description="Events covered in the news report.")
#     quotes: Optional[ExtractedDict] = Field(description="Quotes related to the news, with keys as citation sources and values as the quoted content.")
#     viewpoints: Optional[ExtractedStringList] = Field(description="Different viewpoints regarding the news.")

class NewsReport_conf(BaseModel):
    title: ExtractedString = Field(
        description="The title or headline of the news report.",
        metadata={"free_text": False}
    )
    summary: ExtractedString = Field(
        description="A brief summary of the news report.",
        metadata={"free_text": True}
    )
    publication_date: Optional[ExtractedString] = Field(
        description="The publication date of the report.",
        metadata={"free_text": False}
    )
    keywords: Optional[ExtractedStringList] = Field(
        description="List of keywords or topics covered in the news report.",
        metadata={"free_text": False}
    )
    # quotes: Optional[ExtractedDict] = Field(
    #     description="Quotes related to the news.",
    #     metadata={"free_text": True}
    # )
    viewpoints: Optional[ExtractedStringList] = Field(
        description="Different viewpoints regarding the news.",
        metadata={"free_text": True}
    )

class NewsReport(BaseModel):
    title: str = Field(description="The title or headline of the news report")
    summary: str = Field(description="A brief summary of the news report")
    publication_date: Optional[str] = Field(description="The publication date of the report")
    keywords: Optional[List[str]] = Field(description="List of keywords or topics covered in the news report")
    #events: List[Event] = Field(description="Events covered in the news report")
    #quotes: Optional[dict] = Field(default=None, description="Quotes related to the news, with keys as the citation sources and values as the quoted content. ")
    viewpoints: Optional[List[str]] = Field(default=None, description="Different viewpoints regarding the news")

# --------- You can customize new extraction schemas below -------- #
class ChemicalSubstance(BaseModel):
    name: str = Field(description="Name of the chemical substance")
    formula: str = Field(description="Molecular formula")
    appearance: str = Field(description="Physical appearance")
    uses: List[str] = Field(description="Primary uses")
    hazards: str = Field(description="Hazard classification")

class ChemicalList(BaseModel):
  chemicals: List[ChemicalSubstance] = Field(description="List of chemicals")


class FoundationModels(BaseModel):
    model_id: str = Field(
        description="Unique identifier of the model", metadata={"free_text": True}
    )
    model_name: str = Field(
        description="Only the name of the model without any extra descriptions", metadata={"free_text": True}
    )
    version: str = Field(
        description="Version identifier of the model", metadata={"free_text": True}
    )
    release_date: date = Field(
        description="Release date of the model."
    )
    last_updated: date = Field(
        description="Last updated date of the model."
    )
    short_description: str = Field(
        description="Short summary describing the model", metadata={"free_text": True}
    )
    paper_link: HttpUrl = Field(
        description="URL to the associated publication"
    )
    citations: int = Field(
        description="Number of citations the model has received"
    )
    repository: HttpUrl = Field(
        description="URL to the code repository"
    )
    weights: HttpUrl = Field(
        description="URL to pretrained model weights"
    )
    backbone: str = Field(
        description="Specific backbone used in the architecture", metadata={"free_text": True}
    )
    num_layers: int = Field(
        description="Number of layers in the model"
    )
    num_parameters: float = Field(
        description="Model size in millions of parameters"
    )
    pretext_training_type: str = Field(
        description="Type of pretext training strategy used", metadata={"free_text": True}
    )
    masking_strategy: str = Field(
        description="Masking strategy applied during training", metadata={"free_text": True}
    )
    pretraining: str = Field(
        description="Description of the pretraining approach", metadata={"free_text": True}
    )
    domain_knowledge: List[str] = Field(
        description="List of domain-specific knowledge or methods incorporated"
    )
    backbone_modifications: List[str] = Field(
        description="List of modifications made to the backbone"
    )
    supported_sensors: List[str] = Field(
        description="List of satellite sensors supported"
    )
    modality_integration_type: str = Field(
        description="Integration type of the model, e.g., 'Unimodal', 'Homogeneous Multimodal', 'Heterogeneous Multimodal', or 'Cross-domain Multimodal'.", metadata={"free_text": True}
    )   
    modalities: List[str] = Field(
        description="List of input data modalities the model supports, e.g., 'Multispectral', 'Hyperspectral', 'SAR', 'LiDAR', 'Thermal Infrared', 'Text', etc.", metadata={"free_text": True}
    )
    spectral_alignment: Literal["full", "partial", "none"] = Field(
        description="Whether the model explicitly models spectral continuity and alignment across bands"
    )
    temporal_alignment: Literal["full", "partial", "none"] = Field(
        description="Whether the model explicitly models temporal sequences and alignment of multi-temporal imagery"
    )
    spatial_resolution: str = Field(
        description="Spatial resolution of the data (e.g., '10m', 'variable')", metadata={"free_text": True}
    )
    temporal_resolution: str = Field(
        description="Temporal resolution of the data (e.g., 'variable', revisit frequency)", metadata={"free_text": True}
    )
    bands: List[str] = Field(
        description="List of spectral bands used"
    )

    class PretrainingPhase(BaseModel):
        dataset: str = Field(
            description="Dataset used for pretraining", metadata={"free_text": True}
        )
        regions_coverage: List[str] = Field(
            description="List of geographical regions covered during pretraining"
        )
        time_range: str = Field(
            description="Time range of pretraining data.", metadata={"free_text": True}
        )
        num_images: int = Field(
            description="Number of images used"
        )
        token_size: str = Field(
            description="Token size for model input", metadata={"free_text": True}
        )
        image_resolution: str = Field(
            description="Input image resolution", metadata={"free_text": True}
        )
        epochs: int = Field(
            description="Number of training epochs"
        )
        batch_size: int = Field(
            description="Batch size used during training"
        )
        learning_rate: str = Field(
            description="Learning rate", metadata={"free_text": True}
        )
        augmentations: List[str] = Field(
            description="List of augmentations applied during training"
        )
        processing: List[str] = Field(
            description="List of additional preprocessing steps"
        )
        sampling: str = Field(
            description="Sampling strategy", metadata={"free_text": True}
        )
        processing_level: str = Field(
            description="Processing level of the dataset (e.g., L1C, L2A)", metadata={"free_text": True}
        )
        cloud_cover: str = Field(
            description="Cloud cover filtering strategy", metadata={"free_text": True}
        )
        missing_data: str = Field(
            description="How missing data was handled", metadata={"free_text": True}
        )
        masking_ratio: float = Field(
            description="Masking ratio applied during pretraining"
        )

    pretraining_phases: List[PretrainingPhase] = Field(
        description="List of pretraining phases and their configurations"
    )

    class Benchmark(BaseModel):
        task: str = Field(
            description="Type of task evaluated (e.g., classification, segmentation)", metadata={"free_text": True}
        )
        application: str = Field(
            description="Specific application domain", metadata={"free_text": True}
        )
        dataset: str = Field(
            description="Benchmark dataset name", metadata={"free_text": True}
        )
        metrics: List[str] = Field(
            description="List of metrics used for evaluation"
        )
        metrics_value: List[float] = Field(
            description="A list with only numeric values for each evaluation metric, no additional description"
        )
        sensor: List[str] = Field(
            description="List of sensors used for the benchmark"
        )
        regions: List[str] = Field(
            description="List of regions evaluated"
        )
        original_samples: int = Field(
            description="The total number of samples available in the original dataset before any sampling or filtering"
        )
        num_samples: int = Field(
            description="The actual number of samples used, after applying the sampling percentage to the original dataset"
        )
        sampling_percentage: float = Field(
            description="The fraction of the original dataset retained for modeling, expressed as a percentage (0–100)"
        )
        num_classes: int = Field(
            description="Number of classes in the task"
        )
        classes: List[str] = Field(
            description="List of descriptions/names of each class in the task"
        )
        image_resolution: str = Field(
            description="Input image resolution", metadata={"free_text": True}
        )
        spatial_resolution: str = Field(
            description="Spatial resolution of the data", metadata={"free_text": True}
        )
        bands_used: List[str] = Field(
            description="List of spectral bands used in evaluation"
        )
        augmentations: List[str] = Field(
            description="List of data augmentations applied during evaluation"
        )
        optimizer: str = Field(
            description="Optimizer used for training", metadata={"free_text": True}
        )
        batch_size: int = Field(
            description="Batch size used"
        )
        learning_rate: float = Field(
            description="Learning rate used"
        )
        epochs: int = Field(
            description="Number of epochs run"
        )
        loss_function: str = Field(
            description="Loss function used", metadata={"free_text": True}
        )
        split_ratio: str = Field(
            description="Train/val/test split ratio", metadata={"free_text": True}
        )

    benchmarks: List[Benchmark] = Field(
        description="List of benchmarks evaluating the model."
    )


class FoundationModels_conf(BaseModel):
    model_id: ExtractedString = Field(
        description="Unique identifier of the model.",
        metadata={"free_text": True}
    )
    model_name: ExtractedString = Field(
        description="Only the name of the model without any extra descriptions",
        metadata={"free_text": False}
    )
    version: ExtractedString = Field(
        description="Version identifier of the model.",
        metadata={"free_text": True}
    )
    release_date: ExtractedString = Field(
        description="Release date of the model.",
        metadata={"free_text": False}
    )
    last_updated: ExtractedString = Field(
        description="Last updated date of the model.",
        metadata={"free_text": False}
    )
    short_description: ExtractedString = Field(
        description="Short summary describing the model.",
        metadata={"free_text": True}
    )
    paper_link: ExtractedString = Field(
        description="URL to the associated publication.",
        metadata={"free_text": False}
    )
    citations: ExtractedNumber = Field(
        description="Number of citations the model has received.",
        metadata={"free_text": False}
    )
    repository: ExtractedString = Field(
        description="URL to the code repository.",
        metadata={"free_text": False}
    )
    weights: ExtractedString = Field(
        description="URL to pretrained model weights.",
        metadata={"free_text": False}
    )
    backbone: ExtractedString = Field(
        description="Specific backbone used in the architecture",
        metadata={"free_text": True}
    )
    num_layers: ExtractedNumber = Field(
        description="Number of layers in the model.",
        metadata={"free_text": False}
    )
    num_parameters: ExtractedNumber = Field(
        description="Model size in millions of parameters",
        metadata={"free_text": False}
    )
    pretext_training_type: ExtractedString = Field(
        description="Type of pretext training strategy used",
        metadata={"free_text": True}
    )
    masking_strategy: ExtractedString = Field(
        description="Masking strategy applied during training",
        metadata={"free_text": True}
    )
    pretraining: ExtractedString = Field(
        description="Description of pretraining approach.",
        metadata={"free_text": True}
    )
    domain_knowledge: ExtractedStringList = Field(
        description="List of domain-specific knowledge or methods incorporated"
    )
    backbone_modifications: ExtractedStringList = Field(
        description="List of modifications made to the backbone"
    )
    supported_sensors: ExtractedStringList = Field(
        description="List of satellite sensors supported"
    )
    modality_integration_type: ExtractedString = Field(
        description="Integration type of the model, e.g., 'Unimodal', 'Homogeneous Multimodal', 'Heterogeneous Multimodal', or 'Cross-domain Multimodal'.", metadata={"free_text": True}
    )   
    modalities: ExtractedStringList = Field(
        description="List of input data modalities the model supports, e.g., 'Multispectral', 'Hyperspectral', 'SAR', 'LiDAR', 'Thermal Infrared', 'Text', etc.", metadata={"free_text": True}
    )
    spectral_alignment: ExtractedAlignment = Field(
        description="Whether the model explicitly models spectral continuity and alignment across bands"
    )
    temporal_alignment: ExtractedAlignment = Field(
        description="Whether the model explicitly models temporal sequences and alignment of multi-temporal imagery"
    )
    spatial_resolution: ExtractedString = Field(
        description="Spatial resolution of the data (e.g., '10m', 'variable')",
        metadata={"free_text": True}
    )
    temporal_resolution: ExtractedString = Field(
        description="Temporal resolution of the data (e.g., 'variable', revisit frequency)",
        metadata={"free_text": True}
    )
    bands: ExtractedStringList = Field(
        description="List of spectral bands used"
    )

    class PretrainingPhase(BaseModel):
        dataset: ExtractedString = Field(
            description="Dataset used for pretraining.",
            metadata={"free_text": True}
        )
        regions_coverage: ExtractedStringList = Field(
            description="List of geographical regions covered during pretraining"
        )
        time_range: ExtractedString = Field(
            description="Time range of pretraining data.",
            metadata={"free_text": True}
        )
        num_images: ExtractedNumber = Field(
            description="Number of images used.",
            metadata={"free_text": False}
        )
        token_size: ExtractedString = Field(
            description="Token size for model input",
            metadata={"free_text": True}
        )
        image_resolution: ExtractedString = Field(
            description="Input image resolution",
            metadata={"free_text": True}
        )
        epochs: ExtractedNumber = Field(
            description="Number of training epochs.",
            metadata={"free_text": False}
        )
        batch_size: ExtractedNumber = Field(
            description="Batch size used during training",
            metadata={"free_text": False}
        )
        learning_rate: ExtractedString = Field(
            description="Learning rate.",
            metadata={"free_text": False}
        )
        augmentations: ExtractedStringList = Field(
            description="List of augmentations applied during training"
        )
        processing: ExtractedStringList = Field(
            description="List of additional preprocessing steps"
        )
        sampling: ExtractedString = Field(
            description="Sampling strategy.",
            metadata={"free_text": True}
        )
        processing_level: ExtractedString = Field(
            description="Processing level of the dataset (e.g., L1C, L2A)",
            metadata={"free_text": True}
        )
        cloud_cover: ExtractedString = Field(
            description="Cloud cover filtering strateg",
            metadata={"free_text": True}
        )
        missing_data: ExtractedString = Field(
            description="How missing data was handled",
            metadata={"free_text": True}
        )
        masking_ratio: ExtractedNumber = Field(
            description="Masking ratio applied during pretraining",
            metadata={"free_text": False}
        )

    pretraining_phases: List[PretrainingPhase] = Field(
        description="List of pretraining phases and their configurations"
    )

    class Benchmark(BaseModel):
        task: ExtractedString = Field(
            description="Type of task evaluated (e.g., classification, segmentation)",
            metadata={"free_text": True}
        )
        application: ExtractedString = Field(
            description="Specific application domain",
            metadata={"free_text": True}
        )
        dataset: ExtractedString = Field(
            description="Benchmark dataset name.",
            metadata={"free_text": True}
        )
        metrics: ExtractedStringList = Field(
            description="List of metrics used for evaluation"
        )
        metrics_value: ExtractedNumberList = Field(
            description="A list with only numeric values for each evaluation metric, no additional description",
            metadata={"free_text": False}
        )
        sensor: ExtractedStringList = Field(
            description="List of sensors used for the benchmark"
        )
        regions: ExtractedStringList = Field(
            description="List of regions evaluated"
        )
        original_samples: ExtractedNumber = Field(
            description="The total number of samples available in the original dataset before any sampling or filtering",
            metadata={"free_text": False}
        )
        num_samples: ExtractedNumber = Field(
            description="The actual number of samples used, after applying the sampling percentage to the original dataset",
            metadata={"free_text": False}
        )
        sampling_percentage: ExtractedNumber = Field(
            description="The fraction of the original dataset retained for modeling, expressed as a percentage (0–100)",
            metadata={"free_text": False}
        )
        num_classes: ExtractedNumber = Field(
            description="Number of classes in the task",
            metadata={"free_text": False}
        )
        classes: ExtractedStringList = Field(
            description="List of descriptions/names of each class in the task"
        )
        image_resolution: ExtractedString = Field(
            description="Input image resolution",
            metadata={"free_text": True}
        )
        spatial_resolution: ExtractedString = Field(
            description="Spatial resolution of the data",
            metadata={"free_text": True}
        )
        bands_used: ExtractedStringList = Field(
            description="List of spectral bands used in evaluation"
        )
        augmentations: ExtractedStringList = Field(
            description="List of data augmentations applied during evaluation"
        )
        optimizer: ExtractedString = Field(
            description="Optimizer used for training",
            metadata={"free_text": True}
        )
        batch_size: ExtractedNumber = Field(
            description="Batch size used",
            metadata={"free_text": False}
        )
        learning_rate: ExtractedNumber = Field(
            description="Learning rate used",
            metadata={"free_text": False}
        )
        epochs: ExtractedNumber = Field(
            description="Number of epochs run",
            metadata={"free_text": False}
        )
        loss_function: ExtractedString = Field(
            description="Loss function used",
            metadata={"free_text": True}
        )
        split_ratio: ExtractedString = Field(
            description="Train/val/test split ratio",
            metadata={"free_text": True}
        )

    benchmarks: List[Benchmark] = Field(
        description="List of benchmarks evaluating the model."
    )

# class FoundationModels(BaseModel):
#     # 1. Basic Information
#     model_id: str = Field(description="Unique identifier of the model")
#     model_name: str = Field(description="Name of the model")
#     version: str = Field(description="Model version")
#     release_date: date = Field(description="Release date of the model")
#     last_updated: Optional[date] = Field(default=None, description="Last update date")
#     short_description: str = Field(description="Short description of the model")
#     paper_link: Optional[HttpUrl] = Field(default=None, description="Link to paper")
#     citations: Optional[int] = Field(default=None, description="Citation count")
#     repository: Optional[HttpUrl] = Field(default=None, description="GitHub repository")
#     weights: Optional[HttpUrl] = Field(default=None, description="Pretrained weights URL")

#     # 2. Model Architecture
#     base_type: str = Field(description="Model base type (e.g., vit, cnn)")
#     backbone: str = Field(description="Backbone architecture")
#     num_layers: int = Field(description="Number of layers")
#     pretext_training: Dict[str, str] = Field(description="Type and strategy for pretext training")
#     pretraining: str = Field(description="Pretraining strategy (e.g., self_supervised)")
#     domain_knowledge: List[str] = Field(default=[], description="Domain-specific knowledge applied")
#     backbone_modifications: List[str] = Field(description="Modifications to the backbone")

#     # 3. Input Requirements
#     supported_sensors: List[str] = Field(description="Supported remote sensing sensors")
#     modalities: List[str] = Field(description="Input data modalities")
#     alignment: Dict[str, bool] = Field(description="Alignment across spectral/temporal axes")
#     resolution: Dict[str, Optional[str]] = Field(description="Spatial and temporal resolutions")
#     bands: Dict[str, List[str]] = Field(description="Required and optional bands")

#     # 4. Pretraining Phases
#     class PretrainingPhase(BaseModel):
#         dataset: str = Field(description="Dataset used for pretraining")
#         coverage: Dict[str, List[str]] = Field(description="Geographical and temporal coverage")
#         num_images: int = Field(description="Number of training images")
#         token_size: str = Field(description="Token dimensions (e.g., 8x8x3)")
#         image_resolution: str = Field(description="Input image resolution (e.g., 96x96)")
#         epochs: Optional[int] = Field(description="Training epochs")
#         batch_size: int = Field(description="Batch size used during training")
#         learning_rate: str = Field(description="Learning rate")
#         augmentations: List[str] = Field(description="Data augmentations applied")
#         processing: List[str] = Field(description="Preprocessing steps")
#         sampling: str = Field(description="Sampling strategy")
#         processing_level: str = Field(description="Processing level (e.g., L2A)")
#         cloud_cover: Optional[str] = Field(description="Cloud cover filtering info")
#         missing_data: Optional[str] = Field(description="Handling of missing data")
#         masking_ratio: float = Field(description="Masking ratio used")

#     pretraining_phases: List[PretrainingPhase] = Field(description="List of pretraining settings")

#     # 5. Benchmarks
#     class Benchmark(BaseModel):
#         application: str = Field(description="Application type (e.g., land use classification)")
#         dataset: str = Field(description="Dataset used")
#         metrics: Dict[str, float] = Field(description="Evaluation metrics")
#         sensor: List[str] = Field(description="Sensor(s) used")
#         regions: List[str] = Field(description="Geographical regions")
#         num_samples: Optional[int] = Field(description="Number of samples")
#         classes: int = Field(description="Number of classes")
#         image_resolution: str = Field(description="Input image size")
#         spatial_resolution: Union[str, List[str]] = Field(description="Spatial resolution(s)")
#         bands_used: List[str] = Field(description="Spectral bands used")
#         augmentations: List[str] = Field(description="Augmentations used during benchmarking")
#         optimizer: str = Field(description="Optimizer used")
#         batch_size: int = Field(description="Batch size")
#         learning_rate: str = Field(description="Learning rate")
#         epochs: Optional[int] = Field(description="Number of training epochs")
#         loss_function: str = Field(description="Loss function")
#         split_ratio: str = Field(description="Training/validation/test split info")

#     benchmarks: List[Benchmark] = Field(description="Benchmark evaluations of the model")
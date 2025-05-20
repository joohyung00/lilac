
from enum import Enum, unique


def add_enum_utilities(cls):
    # Add .from_value() classmethod
    @classmethod
    def from_value(cls_, value: str):
        if value == None:
            return None
        for member in cls_:
            if member.value == value:
                return member
        raise ValueError(f"No {cls_.__name__} with value: {value}")
    
    # Add __str__ to return .value directly
    def __str__(self):
        return self.value

    cls.from_value = from_value
    cls.__str__ = __str__
    return cls

@unique
@add_enum_utilities
class Modality(Enum):
    IMAGE = "image"
    TEXT = "text"
    TABLE = "table"

@unique
@add_enum_utilities
class BenchmarkName(Enum):
    MMWEBQA = "MMWebQA"
    M3DOCVQA = "M3DocVQA"
    MMCOQA = "MMCoQA"
    WEBQA = "WebQA"
    
    INFOVQA = "InfoVQA"
    SLIDEVQA = "SlideVQA"
    MPDOCVQA = "MP-DocVQA"
    
@unique
@add_enum_utilities
class BenchmarkType(Enum):
    MULTIMODALQA = "multimodalqa"
    VQA = "vqa"
    
    
@unique
@add_enum_utilities
class AlgorithmName(Enum):
    OMG = "LILaC"
    VISRAG = "VisRAG"
    
@unique
@add_enum_utilities
class LabelType(Enum):
    COMPONENT = "component"
    PAGE = "page"
        
@unique
@add_enum_utilities
class RetrievalMetric(Enum):
    MRR = "MRR@10"
    PAGE_RECALL = "Page Recall"
    COMPONENT_RECALL = "Component Recall"
    DOCUMENT_RECALL = "Document Recall"
    COMPONENT_HIT   = "Component Hit"
    DOCUMENT_HIT      = "Document Hit"
    COMPONENT_MRR     = "Component MRR"
    DOCUMENT_MRR      = "Document MRR" 

@unique
@add_enum_utilities
class EndToEndMetric(Enum):
    EM = "EM"
    F1 = "F1"


@unique 
@add_enum_utilities
class EndToEndMetric(Enum):
    F1 = "F1"
    EM = "EM"
    
    
@unique
@add_enum_utilities
class ComponentIdIdx(Enum):
    DOC_ID = 0
    COMP_ID = 1


def benchmark_name_to_type(benchmark_name: BenchmarkName) -> BenchmarkType:
        
    if benchmark_name in [
        BenchmarkName.MMWEBQA,
        BenchmarkName.MMCOQA,
        BenchmarkName.WEBQA
    ]:
        return BenchmarkType.MULTIMODALQA
    elif benchmark_name in [
        BenchmarkName.INFOVQA,
        BenchmarkName.SLIDEVQA,
        BenchmarkName.MPDOCVQA
    ]:
        return BenchmarkType.VQA
    else:
        raise ValueError(f"Unknown benchmark name: {benchmark_name}")
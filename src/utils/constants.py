
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
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    
    SENTENCE = "sentence"
    PROPOSITION = "proposition"
    
    TABLE_SEGMENT = "table_segment"
    
    SUBIMAGE = "subimage"

    # Class method to get all modalities
    @classmethod
    def get_all_modalities():
        return [
            Modality.TEXT, 
            Modality.TABLE, 
            Modality.IMAGE, 
            Modality.SENTENCE, 
            Modality.PROPOSITION, 
            Modality.TABLE_SEGMENT, 
            Modality.SUBIMAGE
        ]



@unique
@add_enum_utilities
class ParsedWebKeywords(Enum):
    TITLE = "title"
    HIERARCHY = "hierarchy"
    ID_SEQUENCE = "id_sequence"
    ID_TO_COMPONENT = "id_to_component"
    ID_TO_HTML = "id_to_html"
    
    
@unique
@add_enum_utilities
class ModalityIndicator(Enum):
    IMAGE = "i"
    TEXT = "p"
    TABLE = "t"
    HEADER = "h"
    
    
@unique
@add_enum_utilities
class MMDocConstants(Enum):
    COMPONENTS = "COMPONENTS"
    

@unique
@add_enum_utilities
class EmbeddingMode(Enum):
    IMAGE = "image"
    SUMMARY = "summary"
    OCR = "ocr"

    
    
def component_id_to_modality(component_id: str) -> Modality:
    """
    Convert a component ID to its modality.
    
    Args:
        component_id (str): The component ID.
        
    Returns:
        Modality: The modality of the component.
    """
    if component_id.startswith(ModalityIndicator.IMAGE.value):
        return Modality.IMAGE
    elif component_id.startswith(ModalityIndicator.TEXT.value):
        return Modality.TEXT
    elif component_id.startswith(ModalityIndicator.TABLE.value):
        return Modality.TABLE
    else:
        return None
from pydantic import BaseModel

class Banknote(BaseModel):
    variance:float
    skewness:float
    curtosis:float
    entropy:float
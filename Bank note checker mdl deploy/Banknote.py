from pydantic import BaseModel

class Banknote(BaseModel):
    Varience:float
    skewness:float
    curtosis:float
    entropy:float
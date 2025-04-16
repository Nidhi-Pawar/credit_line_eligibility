from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """  Model Configs  """
    model_name : str = "XGBoost"
    fine_tuning: bool = True




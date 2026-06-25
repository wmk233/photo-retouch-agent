from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PhotoAsset(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_id: str = Field(alias="imageId")
    filename: str
    content_type: str = Field(alias="contentType")
    width: int
    height: int
    size_bytes: int = Field(alias="sizeBytes")
    url: str
    created_at: datetime = Field(alias="createdAt")

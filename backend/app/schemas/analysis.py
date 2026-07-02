from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_id: str = Field(alias="imageId")
    domain_type: str = Field(default="portrait", alias="domainType")


class SubjectSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    count: int
    position: str
    face_visibility: str = Field(alias="faceVisibility")


class PhotoAnalysis(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_id: str = Field(alias="imageId")
    domain_type: str = Field(alias="domainType")
    scene_type: str = Field(alias="sceneType")
    subjects: SubjectSummary
    lighting_issues: list[str] = Field(alias="lightingIssues")
    background_issues: list[str] = Field(alias="backgroundIssues")
    portrait_suggestions: list[str] = Field(alias="portraitSuggestions")
    composition_suggestions: list[str] = Field(alias="compositionSuggestions")
    recommended_styles: list[str] = Field(alias="recommendedStyles")
    risk_flags: list[str] = Field(alias="riskFlags")
    brain_provider: str = Field(default="local", alias="brainProvider")
    brain_model: str = Field(default="rule-based-analyzer", alias="brainModel")
    vision_mode: str = Field(default="derived", alias="visionMode")

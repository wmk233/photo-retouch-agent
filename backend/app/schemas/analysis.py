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


class FaceFacts(BaseModel):
    """Observable measurements a stylist could point at in the photo.

    Every field is a coarse heuristic, hedged in the wording, and deliberately
    free of any inferred private attribute (gender, age, ethnicity, occupation,
    personality). See ``app.domains.portrait.facts`` for how they are measured.
    """

    model_config = ConfigDict(populate_by_name=True)

    face_shape: str = Field(default="oval", alias="faceShape")
    face_shape_reason: str = Field(default="未测量", alias="faceShapeReason")
    forehead_height_percent: float = Field(
        default=0.0, alias="foreheadHeightPercent"
    )
    forehead_height_label: str = Field(default="未识别", alias="foreheadHeightLabel")
    face_width_ratio: float = Field(default=0.0, alias="faceWidthRatio")
    face_length_label: str = Field(default="未识别", alias="faceLengthLabel")
    cheek_fullness: str = Field(default="未识别", alias="cheekFullness")
    hair_volume_label: str = Field(default="未识别", alias="hairVolumeLabel")
    hair_coverage_label: str = Field(default="未识别", alias="hairCoverageLabel")
    current_hair_length: str = Field(default="unknown", alias="currentHairLength")
    current_hair_length_label: str = Field(
        default="未识别", alias="currentHairLengthLabel"
    )
    hair_color_name: str = Field(default="未识别", alias="hairColorName")
    hair_texture_label: str = Field(default="未识别", alias="hairTextureLabel")
    wears_glasses: bool = Field(default=False, alias="wearsGlasses")
    glasses_confidence: float = Field(default=0.0, alias="glassesConfidence")
    eyewear_label: str = Field(default="未识别", alias="eyewearLabel")
    skin_tone_rgb: list[int] = Field(default_factory=list, alias="skinToneRgb")
    skin_tone_label: str = Field(default="未识别", alias="skinToneLabel")
    skin_temperature_label: str = Field(
        default="未识别", alias="skinTemperatureLabel"
    )
    upper_body_color_name: str = Field(default="未识别", alias="upperBodyColorName")
    upper_body_color_rgb: list[int] = Field(
        default_factory=list, alias="upperBodyColorRgb"
    )
    upper_body_color_family: str = Field(
        default="未识别", alias="upperBodyColorFamily"
    )
    upper_body_value: str = Field(default="未识别", alias="upperBodyValue")
    upper_body_palette: list[str] = Field(
        default_factory=list, alias="upperBodyPalette"
    )
    upper_body_frame: str = Field(default="未识别", alias="upperBodyFrame")
    frame_type: str = Field(default="未识别", alias="frameType")
    head_shoulder_label: str = Field(default="未识别", alias="headShoulderLabel")
    styling_directions: list[str] = Field(
        default_factory=list, alias="stylingDirections"
    )


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
    face_facts: FaceFacts | None = Field(default=None, alias="faceFacts")
    #: True when the analysis came from the reuse cache instead of a fresh,
    #: billable provider call.
    analysis_reused: bool = Field(default=False, alias="analysisReused")

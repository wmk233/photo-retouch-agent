from app.schemas.analysis import PhotoAnalysis
from app.schemas.retouch import RetouchPlan


class GeneralPlanner:
    domain_type = "general"

    def create_plans(self, analysis: PhotoAnalysis) -> list[RetouchPlan]:
        scene = analysis.scene_type
        return [
            RetouchPlan(
                plan_id="natural",
                domain_type=analysis.domain_type,
                title="自然增强",
                description=f"优化{scene}的光线与色彩，保持真实观感。",
                intensity="natural",
                edit_prompt=(
                    "自然校正曝光、白平衡和色彩层次，增强主体清晰度，"
                    "保持原始内容、材质、结构和空间关系。"
                ),
                negative_prompt=(
                    "避免新增或删除关键主体，避免结构变形、色彩失真、"
                    "过度锐化和明显 AI 生成痕迹。"
                ),
                expected_changes=["曝光校正", "色彩自然", "主体清晰"],
            ),
            RetouchPlan(
                plan_id="clean",
                domain_type=analysis.domain_type,
                title="清晰通透",
                description="整理画面干扰，增强主体和背景层次。",
                intensity="medium",
                edit_prompt=(
                    "提升画面通透度与局部对比，适度清理边缘干扰物，"
                    "保持主要对象、自然纹理和构图不变。"
                ),
                negative_prompt=(
                    "避免改变主体数量和形状，避免假景深、过饱和、"
                    "涂抹感和错误纹理。"
                ),
                expected_changes=["层次提升", "干扰弱化", "细节清晰"],
            ),
            RetouchPlan(
                plan_id="mood",
                domain_type=analysis.domain_type,
                title="氛围风格",
                description="在保留真实内容的前提下强化情绪和色彩表达。",
                intensity="medium",
                edit_prompt=(
                    "根据场景内容建立协调的色彩风格和光影氛围，"
                    "保留主体真实性、环境细节和原始构图。"
                ),
                negative_prompt=(
                    "避免改变场景语义、主体身份和真实结构，"
                    "避免极端色偏、过强滤镜和生成伪影。"
                ),
                expected_changes=["氛围统一", "色彩表达", "光影增强"],
            ),
        ]

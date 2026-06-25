from app.schemas.analysis import PhotoAnalysis
from app.schemas.retouch import RetouchPlan


class PortraitPlanner:
    domain_type = "portrait"

    def create_plans(self, analysis: PhotoAnalysis) -> list[RetouchPlan]:
        return [
            self._plan(
                analysis=analysis,
                plan_id="natural",
                title="自然美化",
                description="轻微提亮，统一肤色和整体质感，保留真实本人感。",
                intensity="natural",
                negative_prompt="避免过度磨皮、避免改变五官、避免塑料质感",
                expected_changes=["提亮", "肤色统一", "轻质感"],
                goal="自然美化，轻微提亮，肤色均匀，不改变五官。",
            ),
            self._plan(
                analysis=analysis,
                plan_id="avatar",
                title="精致头像",
                description="清理视觉干扰，强化面部清晰度，输出更适合头像的构图。",
                intensity="medium",
                negative_prompt="避免强换脸、避免证件照伪造、避免背景脏纹",
                expected_changes=["头像裁切", "背景弱化", "脸部清晰"],
                goal="头像构图，面部清晰，背景干净，主体真实。",
            ),
            self._plan(
                analysis=analysis,
                plan_id="mood",
                title="氛围风格",
                description="增强色彩情绪和层次，保留自然人像基础。",
                intensity="medium",
                negative_prompt="避免色偏过重、避免人像 AI 化、避免皮肤失真",
                expected_changes=["胶片感", "色彩层次", "轻颗粒"],
                goal="胶片/社媒氛围，增强色彩情绪，保留本人感。",
            ),
        ]

    def _plan(
        self,
        analysis: PhotoAnalysis,
        plan_id: str,
        title: str,
        description: str,
        intensity: str,
        negative_prompt: str,
        expected_changes: list[str],
        goal: str,
    ) -> RetouchPlan:
        edit_prompt = "\n".join(
            [
                f"场景：{analysis.scene_type}",
                f"光线处理：{'、'.join(analysis.lighting_issues)}",
                f"背景处理：{'、'.join(analysis.background_issues)}",
                f"人像要求：{'、'.join(analysis.portrait_suggestions)}",
                f"目标：{goal}",
            ]
        )
        return RetouchPlan(
            plan_id=plan_id,
            domain_type=self.domain_type,
            title=title,
            description=description,
            intensity=intensity,
            edit_prompt=edit_prompt,
            negative_prompt=negative_prompt,
            expected_changes=expected_changes,
        )

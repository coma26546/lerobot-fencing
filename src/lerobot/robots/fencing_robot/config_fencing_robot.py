from dataclasses import dataclass, field

from ..config import RobotConfig
from ..fencing_arm.config_fencing_arm import FencingArmConfig
from ..roboclaw import RoboClawDriveConfig


@dataclass
class FencingRobotArmConfig(FencingArmConfig):
    """Limit nested CLI choices to the arm, avoiding recursive robot choices."""

    @classmethod
    def get_known_choices(cls) -> dict:
        return {"fencing_arm": cls}

    @classmethod
    def get_choice_class(cls, name: str):
        if name != "fencing_arm":
            raise ValueError("The combined robot's arm must be fencing_arm")
        return cls

    @classmethod
    def get_choice_name(cls, subclass):
        if not issubclass(subclass, FencingArmConfig):
            raise ValueError("The combined robot's arm must be fencing_arm")
        return "fencing_arm"


@RobotConfig.register_subclass("fencing_robot")
@dataclass(kw_only=True)
class FencingRobotConfig(RobotConfig):
    arm: FencingRobotArmConfig
    drive: RoboClawDriveConfig = field(default_factory=RoboClawDriveConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.drive.port in (self.arm.port_1, self.arm.port_2):
            raise ValueError("The RoboClaw port must differ from both arm ports")

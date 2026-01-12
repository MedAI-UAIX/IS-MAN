from pydantic import BaseModel, Field
from langchain.tools import tool

robot_sdk = RoboticArmAPI()

class ArmControlSchema(BaseModel):
    action: str = Field(..., description="Options: start, end.", examples=["start", "end"])

@tool(args_schema=ArmControlSchema)
def ControlRobotArm(action: str):
    if action == "start":
        res = robot_sdk.start_arm()
    elif action == "end":
        res = robot_sdk.stop_arm()
    else:
        return {"status": "error", "message": "Unknown action."}
    status = "success" if res["status_code"] == 200 else "error"
    return {"status": status, "message": res["message"]}



class ForceControlSchema(BaseModel):
    action: str = Field(..., description="Options: calibrate, up, down.", examples=["calibrate", "up", "down"])

@tool(args_schema=ForceControlSchema)
def ControlForce(action: str):
    if action == "calibrate":
        res = robot_sdk.calibrate_force_sensor()
    elif action == "up":
        res = robot_sdk.adjust_probe_pressure(increase=True)
    elif action == "down":
        res = robot_sdk.adjust_probe_pressure(increase=False)
    else:
        return {"status": "error", "message": "Unknown force control action."}
    status = "success" if res["status_code"] == 200 else "error"
    return {"status": status, "message": res["message"]}



class MotionControlSchema(BaseModel):
    action: str = Field(..., description="Options: pause, stop, speed_up, speed_down.",
                        examples=["pause", "stop", "speed_up", "speed_down"])

@tool(args_schema=MotionControlSchema)
def ControlMotion(action: str):
    res = robot_sdk.motion_command(action)
    status = "success" if res["status_code"] == 200 else "error"
    return {"status": status, "message": res["message"]}



class ControlModeSchema(BaseModel):
    mode: str = Field(..., description="Options: cartesian, impedance, hybrid.",
                      examples=["cartesian", "impedance", "hybrid"])

@tool(args_schema=ControlModeSchema)
def SwitchControlMode(mode: str):
    res = robot_sdk.switch_mode(mode)
    status = "success" if res["status_code"] == 200 else "error"
    return {"status": status, "message": res["message"]}



class KeypointDetectionSchema(BaseModel):
    region: str = Field(..., description="Options: abdomen, neck.", examples=["abdomen", "neck"])

@tool(args_schema=KeypointDetectionSchema)
def DetectKeypoints(region: str):
    res = robot_sdk.detect_keypoints(region)
    if res["status_code"] == 200:
        message = f"{region.capitalize()} keypoints detected: {res['points_detected']} points."
        status = "success"
    else:
        message = "Detection failed."
        status = "error"
    return {"status": status, "message": message}

class UltrasoundScanSchema(BaseModel):
    organ: str = Field(..., description="Options: thyroid, liver, kidney, etc.", examples=["thyroid"])
    region: str = Field(..., description="Scan region", examples=["left_lobe", "right_lobe"])
    direction: str = Field(..., description="Scan direction", examples=["longitudinal", "transverse"])


@tool(args_schema=UltrasoundScanSchema)
def UltrasoundScan(organ: str, region: str, direction: str, agent=None):
    try:
        res = robot_sdk.scan(organ=organ, region=region, direction=direction)
        status = "success" if res["status_code"] == 200 else "error"
        message = res.get("message", "")
    except Exception as e:
        status = "error"
        message = str(e)

    if agent is not None and status == "success":
        if not hasattr(agent, "nodule_detection"):
            agent.nodule_detection = []

        agent.nodule_detection.append({
            "organ": organ,
            "region": region,
            "direction": direction,
            "status": status,
            "message": message
        })

    return {
        "status": status,
        "message": message,
        "scan_info": {
            "organ": organ,
            "region": region,
            "direction": direction
        }
    }
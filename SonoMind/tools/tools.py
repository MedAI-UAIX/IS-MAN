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


class MMOESchema(BaseModel):
    image_path: str = Field(..., description="Path to the thyroid ultrasound image")


@tool(args_schema=MMOESchema)
def MMOEAnalyze(image_path: str):
    """Analyze various TR features of the thyroid."""
    features = mmoe.predict(image_path)  # Returns a dict of TR features
    print(f"[MMOE] Analysis completed: {features}")
    return {"status": "success", "model": "MMOE", "result": features}



class ThynetSchema(BaseModel):
    image_path: str = Field(..., description="Path to the thyroid nodule ultrasound image")


@tool(args_schema=ThynetSchema)
def ThynetClinical(image_path: str):
    """Classify thyroid nodule malignancy (clinical model)."""
    result = thynet.predict(image_path)
    print(f"[Thynet Clinical] Analysis completed: {result}")
    return {"status": "success", "model": "Thynet", "result": result}



class ThynetSSchema(BaseModel):
    image_path: str = Field(..., description="Path to the thyroid nodule ultrasound image")


@tool(args_schema=ThynetSSchema)
def ThynetScreening(image_path: str):
    """Classify thyroid nodule malignancy (screening model)."""
    result = thynet_s.predict(image_path)
    print(f"[Thynet Screening] Analysis completed: {result}")
    return {"status": "success", "model": "Thynet-S", "result": result}



class FollowUpSchema(BaseModel):
    prev_image: str = Field(..., description="Path to the previous ultrasound image")
    curr_image: str = Field(..., description="Path to the current ultrasound image")


@tool(args_schema=FollowUpSchema)
def ThyroidFollowUp(prev_image: str, curr_image: str):
    """Analyze changes between two thyroid ultrasound examinations."""
    result = followup_llm.analyze_change(prev_image, curr_image)
    print(f"[Follow-up LLM] Analysis completed: {result}")
    return {"status": "success", "model": "Follow-up LLM", "result": result}

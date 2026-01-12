import json
from langchain_openai import ChatOpenAI
import os
import sys
from thynet.Hu_Models import Thynet
from thynets.Models4demo import Thynetv2
from mmoe.MMOE_IMG import MMoE
import json

from PIL import Image
import numpy as np
from torchvision import transforms
import torch


class ThynetModel:
    def __init__(self,patient_ID="001"):
        self.model_name = "thynet"
        self.model = Thynet()
        self.model.eval()
        self.patient_ID=patient_ID
    def predict(self):
        img_root = "SonoMind/patient"
        patient_dir = os.path.join(img_root, self.patient_ID)
        if not os.path.isdir(patient_dir):
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
        json_files = [f for f in os.listdir(patient_dir) if f.endswith(".json")]

        if len(json_files) == 0:
            raise FileNotFoundError(f"No json file found in {patient_dir}")
        if len(json_files) > 1:
            raise ValueError(f"Multiple json files found in {patient_dir}: {json_files}")

        json_path = os.path.join(patient_dir, json_files[0])

        with open(json_path, "r", encoding="utf-8") as f:
            nodules_info = json.load(f)

        nodules_results = []
        sentence_list = []

        for nodule in nodules_info:
            nodule_id = nodule["nodule_id"]
            location = nodule["location"]
            img_name = nodule["img_dir"]

            img_path = os.path.join(patient_dir, img_name)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
        
            image = Image.open(img_path)
     
            inputs = transforms.Compose(
                    [transforms.Resize([224,224]),
                     transforms.ToTensor()])(image)
            inputs = inputs.unsqueeze(0)
            with torch.no_grad():
                inputs = inputs
                pred = self.model(inputs)
                if pred.argmax(dim=-1) == 0:
                    assessment = 'benign'
                else:
                    assessment = 'malignant'


                nodules_results.append({
                "nodule_id": nodule_id,
                "location": location,
                "assessment": assessment
                })

                sentence = (
                f"nodule {nodule_id}, location: {location}, "
                f"assessment: {assessment}"
                )
                sentence_list.append(sentence)

        summary_sentence = (
            "Nodule assessments from the Thynet model: "
            + "; ".join(sentence_list)
            + "."
        )

        output = {
            "model": self.model_name,
            "nodules": nodules_results,
            "summary_sentence": summary_sentence
        }
        return output

class ThynetSModel:
    def __init__(self,patient_ID="001"):
        self.model_name = "thynet-s"
        self.model1 = Thynetv2()
        self.model1.load_state_dict(torch.load(r'SonoMind/tools/thynets/Hu_Weigh/combine_v2/node_auc.pth'))
        self.model1.eval()
        self.model2 = Thynetv2()
        self.model2.load_state_dict(torch.load(r'SonoMind/tools/thynets/Hu_Weigh/combine_v2/bm_auc.pth'))
        self.model2.eval()
        self.patient_ID=patient_ID
    def predict(self):
        img_root = "SonoMind/patient"
        patient_dir = os.path.join(img_root, self.patient_ID)
        if not os.path.isdir(patient_dir):
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
        json_files = [f for f in os.listdir(patient_dir) if f.endswith(".json")]

        if len(json_files) == 0:
            raise FileNotFoundError(f"No json file found in {patient_dir}")
        if len(json_files) > 1:
            raise ValueError(f"Multiple json files found in {patient_dir}: {json_files}")

        json_path = os.path.join(patient_dir, json_files[0])

        with open(json_path, "r", encoding="utf-8") as f:
            nodules_info = json.load(f)

        nodules_results = []
        sentence_list = []

        for nodule in nodules_info:
            nodule_id = nodule["nodule_id"]
            location = nodule["location"]
            img_name = nodule["img_dir"]

            img_path = os.path.join(patient_dir, img_name)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
        
            image = Image.open(img_path).convert('RGB')
     
            inputs = transforms.Compose(
                    [transforms.Resize([224,224]),
                     transforms.ToTensor()])(image)
            inputs = inputs.unsqueeze(0)
            with torch.no_grad():
                node_pred, _, _ = self.model1(inputs)
                has_nodule = node_pred.argmax(dim=-1).item()
                prob_has_nodule = node_pred.squeeze()[1].item()
                _, BM_pred, _ = self.model2(inputs)
                malign_pred = BM_pred.argmax(dim=-1).item()
                prob = BM_pred.squeeze()[1].item()
                if has_nodule == 0:
                    assessment = 'no nodule'
                else:
                    assessment = 'malignant' if malign_pred == 1 else 'benign'

                nodules_results.append({
                "nodule_id": nodule_id,
                "location": location,
                "assessment": assessment
                })

                sentence = (
                f"nodule {nodule_id}, location: {location}, "
                f"assessment: {assessment}"
                )
                sentence_list.append(sentence)

        summary_sentence = (
            "Nodule assessments from the Thynet model: "
            + "; ".join(sentence_list)
            + "."
        )

        output = {
            "model": self.model_name,
            "nodules": nodules_results,
            "summary_sentence": summary_sentence
        }
        return output
   
 
class MMOETool:
    def __init__(self,patient_ID="001"):
        self.model_name = "MMOE"
        self.model = MMoE(num_experts=35, num_feature=1024, bottom_mlp_dims=(512, 256, 128), tower_mlp_dims=(128, 64, 32),
                        dropout=0.2, tasks=(6,5,5,2,2,2,2,2,2,2,5))
        weight_path = "SonoMind/tools/throid_TI-RADS/throid_TI-RADS/weight/2-TI_RADS/best_f1.pth"
        self.model.load_state_dict(torch.load(weight_path)['model_state_dict'])
        self.model.eval()
        self.patient_ID=patient_ID
    def predict(self):
        img_root = "SonoMind/patient"
        patient_dir = os.path.join(img_root, self.patient_ID)
        if not os.path.isdir(patient_dir):
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
        json_files = [f for f in os.listdir(patient_dir) if f.endswith(".json")]

        if len(json_files) == 0:
            raise FileNotFoundError(f"No json file found in {patient_dir}")
        if len(json_files) > 1:
            raise ValueError(f"Multiple json files found in {patient_dir}: {json_files}")

        json_path = os.path.join(patient_dir, json_files[0])

        with open(json_path, "r", encoding="utf-8") as f:
            nodules_info = json.load(f)

        nodules_results = []
        sentence_list = []

        for nodule in nodules_info:
            nodule_id = nodule["nodule_id"]
            location = nodule["location"]
            nodule_size = f"{nodule['size_mm']}mm"
            img_name = nodule["img_dir"]
            assessment_parts = []
            img_path = os.path.join(patient_dir, img_name)

            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
        
            image = Image.open(img_path).convert('RGB') 
            val_transforms= transforms.Compose([      
                                            transforms.Resize((128,128)),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            image = val_transforms(image)
            image = image.unsqueeze(0)

            output = self.model(image)

            pred_component=torch.softmax(output[0], -1)[0]
            pred_echoes=torch.softmax(output[1], -1)[0]
            pred_edge=torch.softmax(output[2], -1)[0]
            pred_form=torch.softmax(output[3], -1)[0]
            pred_strongEchoes_No_strong_echoes=torch.softmax(output[4], -1)[0]
            pred_strongEchoes_Comet_tail_artifact=torch.softmax(output[5], -1)[0]
            pred_strongEchoes_Coarse_calcification=torch.softmax(output[6], -1)[0]
            pred_strongEchoes_Peripheral_calcification=torch.softmax(output[7], -1)[0]
            pred_strongEchoes_Echogenic_foci=torch.softmax(output[8], -1)[0]

            component = ['cystic', 'almost completely cystic', 'spongiform', 'mixed cystic and solid', 'solid', 'almost completely solid']
            echoes = ['anechoic', 'hyperechoic', 'isoechoic', 'hypoechoic', 'very hypoechoic']
            edge = ['smooth', 'ill-defined',  'lobulated', 'irregular', 'extra-thyroidal extension']
            form = ['wider-than-tall', 'taller-than-wide']

            index_1 = pred_component.argmax(dim=-1)
            index_2 = pred_echoes.argmax(dim=-1)
            index_3 = pred_edge.argmax(dim=-1)
            index_4 = pred_form.argmax(dim=-1)
            index_5 = pred_strongEchoes_No_strong_echoes.argmax(dim=-1)
            index_6 = pred_strongEchoes_Comet_tail_artifact.argmax(dim=-1)
            index_7 = pred_strongEchoes_Coarse_calcification.argmax(dim=-1)
            index_8 = pred_strongEchoes_Peripheral_calcification.argmax(dim=-1)
            index_9 = pred_strongEchoes_Echogenic_foci.argmax(dim=-1)

            assessment_parts.append(component[index_1])
            assessment_parts.append(echoes[index_2])
            assessment_parts.append(edge[index_3])
            assessment_parts.append(form[index_4])
            if index_5 == 1:
                assessment_parts.append('no echogenic foci')
            if index_6 == 1:
                assessment_parts.append("large comet-tail artifacts")
            if index_7 == 1:
                assessment_parts.append("macrocalcifications")
            if index_8 == 1:
                assessment_parts.append("macrocalcifications")
            if index_9 == 1:
                assessment_parts.append("punctate echogenic foci")

            #TI-RADS
            score = 0
            if index_1 == 0:
                score += 0
            elif index_1 == 1:
                score += 0
            elif index_1 == 2:
                score += 0
            elif index_1 == 3:
                score += 1
            elif index_1 == 4:
                score += 2
            elif index_1 == 5:
                score += 2
            if index_2 == 0:
                score += 0
            elif index_2 == 1:
                score += 1
            elif index_2 == 2:
                score += 1
            elif index_2 == 3:
                score += 2
            elif index_2 == 4:
                score += 3
            if index_3 == 0:
                score += 0
            elif index_3 == 1:
                score += 0
            elif index_3 == 2:
                score += 2
            elif index_3 == 3:
                score += 2
            elif index_3 == 4:
                score += 3
            if index_4 == 0:
                score += 0
            elif index_4 == 1:
                score += 3
            if index_7 == 0:
                score += 0
            elif index_7 == 1:
                score += 1
            if index_8 == 0:
                score += 0
            elif index_8 == 1:
                score += 2
            if index_9 == 0:
                score += 0
            elif index_9 == 1:
                score += 3
            if score == 0:
                assessment_parts.append("TR1")
            elif score <= 2:
                assessment_parts.append("TR2")
            elif score == 3:
                assessment_parts.append("TR3")
            elif 4<= score <=6:
                assessment_parts.append("TR4")
            elif score >= 7:
                assessment_parts.append("TR5")

            assessment = "; ".join(assessment_parts)
            nodules_results.append({
            "nodule_id": nodule_id,
            "location": location,
            "nodule_size": nodule_size,
            "assessment": assessment
            })

            sentence = (
            f"nodule {nodule_id}, location: {location}, nodule size: {nodule_size}"
            f"assessment: {assessment}"
            )
            sentence_list.append(sentence)

        summary_sentence = (
            "Nodule assessments from the Thynet model: "
            + "; ".join(sentence_list)
            + "."
        )

        output = {
            "model": self.model_name,
            "nodules": nodules_results,
            "summary_sentence": summary_sentence
        }
        return output


class FollowUpTool:
    def __init__(self,patient_ID="001"):
        self.model_name = "FollowUp"
        API_KEY = "..."
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model = "gemini-2.5-flash"
        self.llm = ChatOpenAI(model=model, api_key=API_KEY, base_url=base_url)
        self.patient_ID=patient_ID
        self.mmoe_tool = MMOETool(patient_ID=patient_ID)


    def run_follow_up(self, prev_report, surgery_history):

        mmoe_output = self.mmoe_tool.predict()["summary_sentence"]

        follow_up_prompt = f"""
        You are a professional ultrasound radiologist.
        Below are the results from a patient's first and most recent thyroid ultrasound examinations. 
        Please extract key findings from the first report and compare them with the second report.  
        Provide a concise summary describing interval changes in the lesion(s), including nodule number, size, composition, echogenicity, shape, margin, and presence of punctate echogenic foci.

        First examination report:
        {prev_report}

        Second examination report:
        Surgical history: {surgery_history}

        Nodule features:
        {json.dumps(mmoe_output, indent=2)}
"""

        llm_output = self.llm.invoke([{
            "role": "user",
            "content": follow_up_prompt
        }]).content

        return {
            "tool": "follow-up LLM",
            "result":llm_output
        }
    
if __name__ == '__main__':
    tool = MMOETool()
    print(tool.predict()) 
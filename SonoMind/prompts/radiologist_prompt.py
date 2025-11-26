system_prompt = """
You are a Radiologist Agent specializing in ultrasound image interpretation, particularly in the assessment of thyroid nodules.
You are provided with a set of tools and must choose the appropriate tool when necessary.
Your Core Responsibilities:
1.Image Analysis: When a task involves determining the benignity or malignancy of thyroid nodules, or follow-up comparison of ultrasound images, you should invoke the appropriate analysis tool.
2.TI-RADS Grading: When required to determine the TI-RADS category, infer the grade based on the nodule’s imaging features following the ACR-TIRADS guidelines step by step.
3.Report Generation: When tasked with report generation, you must first call the MMOE model to analyze the nodule features, and then use those analyzed features to compose the final report.
4.After completing the report generation, transfer the result back to the supervisor using the transfer_to_supervisor tool.

Thyroid Ultrasound Report Template:
Ultrasound Description:
Describe the general echotexture of the thyroid parenchyma.
For each lobe (left, right) and the isthmus, report:
Presence of nodules or masses (single or multiple).
Characteristics of each nodule.
Provide the diameter of the nodule; if multiple nodules are present, indicate the range of diameters.
Ultrasound Conclusion:
Provide a diagnostic impression.
Assign an appropriate ACR TI-RADS classification if nodule(s) are identified (e.g., “TI-RADS 2”).

Available Tools:
{
  "name": "ImageAnalysis",
  "description": "Perform analysis on thyroid ultrasound images.",
  "parameters": {
    "properties": {
      "model": {
        "description": "Specify the diagnostic model to use. 'thynet' is for clinical diagnostic scenarios, 'thynet-s' is for screening scenarios, 'follow-up LLM' is for follow-up comparison, and 'MMOE' is for TI-RADS feature analysis.",
        "type": "string"
      }
    },
    "required": ["model"]
  }
}
{
  "name": "transfer_to_supervisor",
  "description": "Transfer the current task or conversation context to the supervisor agent for further handling.",
  "parameters": {
    "properties": {
      "input": {
        "description": "A brief summary of the results.",
        "type": "string"
      }
    }
  }
}

Output your response strictly in the following JSON format, without adding any characters, punctuation, or text before or after:
{
  "content": "Ultrasound report or your message to the patient or system",
  "tool_calls": {
    "name": "ToolName",
    "arguments": {
      "parameter": ...
    }
  }
}
"""

input_output = """
Below is the description of the thyroid nodule features. Please calculate the ACR TI-RADS score and the corresponding ACR TI-RADS level.
"""

cot = """
Below is the description of the thyroid nodule features. Please calculate the ACR TI-RADS score and the corresponding ACR TI-RADS level.
First, scoring each feature according to the ACR TI-RADS Scoring Criteria:
The scoring system involves five major ultrasound features:
    Composition (choose 1):
       Cystic or almost completely cystic = 0 points
       Spongiform = 0 points
       Mixed cystic and solid = 1 point
Solid or almost completely solid = 2 points
Echogenicity (choose 1):
Anechoic = 0 points
Hyperechoic or isoechoic = 1 point
Hypoechoic = 2 points
Very Hypoechoic = 3 points
    Shape (choose 1):
Wider-than-tall = 0 points
Taller-than-wide = 3 points
    Margin (choose 1):
Smooth = 0 points
Ill-defined = points
Lobulated or irregular = 2 points
Extra-thyroidal extension = 3points
    Echogenic Foci (choose all that apply):
None or large comet-tail artifacts = 0 points
Macrocalcifications = 1 point
Peripheral (rim) calcifications = 2 points
		Punctate echogenic foci = 3 points
Then sum up the points to give a total score. The total score is used to assign the TI-RADS level, which indicates the likelihood of malignancy and guides further management.
TI-RADS Levels Based on Total Score:
    TI-RADS 1 (0 points): Benign
    TI-RADS 2 (2 points): Not Suspicious
    TI-RADS 3 (3 points): Mildly suspicious
    TI-RADS 4 (4-6 points): Moderately suspicious
TI-RADS 5 (7 points or more): Highly suspicious
"""

icl = """
Below is the description of the thyroid nodule features. Please calculate the ACR TI-RADS score and the corresponding ACR TI-RADS level. 
Example 1:
Characteristics of thyroid nodule: Cystic, Anechoic, Wider-than-tall, Smooth, Large comet-tail artifacts
1.	Composition: Cystic = 0 point
2.	Echogenicity: Anechoic = 0 point
3.	Shape: Wider-than-tall = 0 point
4.	Margin: Smooth = 0 point
5.	Echogenic Foci: Large comet-tail artifacts = 0 point
Total TI-RADS Score: 0 (Composition) + 0 (Echogenicity) + 0 (Shape) + 0 (Margin) + 0 (Echogenic Foci) = 0 points
TI-RADS Level: TI-RADS 1 (0 points)

Example 2:
Characteristics of thyroid nodule: Mixed cystic and solid, Isoechoic, Wider-than-tall, Smooth, No strong echoes.
1.	Composition: Mixed cystic and solid = 1 point
2.	Echogenicity: Isoechoic = 1 point
3.	Shape: Wider-than-tall = 0 point
4.	Margin: Smooth = 0 point
5.	Echogenic Foci: No strong echoes = 0 point
Total TI-RADS Score: 1 (Composition) + 1 (Echogenicity) + 0 (Shape) + 0 (Margin) + 0 (Echogenic Foci) = 2 points
TI-RADS Level: TI-RADS 2 (2 points)

Example 3:
Characteristics of thyroid nodule: Solid, Hyperechoic, Wider-than-tall, Smooth, No strong echoes
1. Composition: Solid or almost completely solid = 2 points
2. Echogenicity: Hyperechoic = 1 point
3. Shape: Wider-than-tall = 0 points
4. Margin: Smooth = 0 points
5. Echogenic Foci: None or large comet-tail artifacts = 0 points
Total TI-RADS Score: 2 (Composition) + 1 (Echogenicity) + 0 (Shape) + 0 (Margin) + 0 (Echogenic Foci) = 3 points
TI-RADS Level: TI-RADS 3 (3 points)

Example 4:
Characteristics of thyroid nodule: Solid, Anechoic, Wider-than-tall, Irregular, No strong echoes
1.	Composition: Solid = 2 point
2.	Echogenicity: Anechoic = 0 point
3.	Shape: Wider-than-tall = 0 point
4.	Margin: Irregular = 2 point
5.	Echogenic Foci: No strong echoes = 0 point
Total TI-RADS Score: 2 (Composition) + 0 (Echogenicity) + 0 (Shape) + 2 (Margin) + 0 (Echogenic Foci) = 4 points
TI-RADS Level: TI-RADS 4 (4 points)

Example 5:
Characteristics of thyroid nodule: Solid, Hypoechoic, Taller-than-wide, Irregular, Punctate echogenic foci, Macrocalcifications
1. Composition: Solid or almost completely solid = 2 points
2. Echogenicity: Hypoechoic = 2 points
3. Shape: Taller-than-wide = 3 points
4. Margin: Irregular = 2 points
5. Echogenic Foci:
Punctate echogenic foci = 3 points
Macrocalcifications = 1 point
Total TI-RADS Score: 2 (Composition) + 2 (Echogenicity) + 3 (Shape) + 2 (Margin) + (3 + 1) (Echogenic Foci) = 13 points
TI-RADS Level: TI-RADS 5 (7 points or more) 
"""
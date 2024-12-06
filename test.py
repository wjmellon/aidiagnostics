import os
import csv
import logging
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ResponseClassifier:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env or pass directly.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # EXACT criteria as provided
        self.criteria = """
        Evaluation Criteria:
        1. Evidence is supplied via citation is relevant to the question and answer 25% 
        Is there a citation at the end of the response? If yes, then this is a 1, else 0

        2. Up to date information (within last 10 years)  20%
        Is the evidence that is used within the last 10 years of the current date? (if yes, then 1, else 0)

        3. Directly answers the question without straying off-topic (compare to query topic)15%
        Compare the question with the answer and if the answer contains words found in the question, at a high frequency (60%) then 1, else 0

        4. The answer is clear and not overly complex (avoid high level vocab or abbreviations) 13%
        If there are abbreviations or scientific jargon related to skin cancer that is used in the field, then mark this a 0, else a 1

        5. Addresses every important aspect of the question 10%
        Are there multiple questions asked in the question? If so, if the answer answers all of the questions mark this a 1, else mark it a 0.
        """

    def classify_response(self, question: str, response: str) -> Dict[str, float]:
        try:
            chat_response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert evaluator scoring responses exactly according to the given criteria."
                    },
                    {
                        "role": "user", 
                        "content": f"""
                            Evaluate this response based on the EXACT criteria below:

                            Question: {question}
                            Response: {response}

                            {self.criteria}

                            IMPORTANT:
                            - Provide ONLY a JSON object with keys:
                            1. "Citation"
                            2. "Recency"
                            3. "TopicAlignment"
                            4. "Clarity"
                            5. "Comprehensiveness"
                            - Each value should be either 0 or 1.

                            For each criterion, explain the reasoning behind the grade in detail. Use plain text after the JSON object for these explanations.
                            Example response format:
                            {{ "Citation": 1, "Recency": 0, "TopicAlignment": 1, "Clarity": 0, "Comprehensiveness": 1 }}

                            Reasoning:

                            Citation: The response contains a proper citation at the end, meeting the requirement.
                            Recency: The evidence provided is older than 10 years, so it does not meet the criterion.
                            TopicAlignment: The answer directly addresses the question and uses relevant terminology, earning a grade of 1.
                            Clarity: The response includes technical jargon and abbreviations without explanation, making it less clear.
                            Comprehensiveness: All aspects of the question are addressed in the response, so this criterion is fulfilled.
                        """
                    }
                ],
                response_format={"type": "text"},
                max_tokens=500
            )
            
            raw_response = chat_response.choices[0].message.content
            json_part, reasoning = raw_response.split("\n\nReasoning:", 1)
            classification = json.loads(json_part.strip())
            
            # Attach reasoning as part of the classification for further inspection if needed
            classification["reasoning"] = reasoning.strip()
            return classification
        
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {"error": str(e)}
    
    
    def process_csv(self, input_file: str, output_file: str) -> None:
        try:
            df = pd.read_csv(input_file)
            
            if len(df.columns) < 5:
                raise ValueError("CSV must have at least 5 columns: question, LLM_1, LLM_2, LLM_3, LLM_4")
            
            # Evaluate each LLM response
            df['LLM_1_Classification'] = df.apply(
                lambda row: self.classify_response(row['question'], row['LLM_1']), 
                axis=1
            )
            df['LLM_2_Classification'] = df.apply(
                lambda row: self.classify_response(row['question'], row['LLM_2']), 
                axis=1
            )
            df['LLM_3_Classification'] = df.apply(
                lambda row: self.classify_response(row['question'], row['LLM_3']), 
                axis=1
            )
            df['LLM_4_Classification'] = df.apply(
                lambda row: self.classify_response(row['question'], row['LLM_4']), 
                axis=1
            )
            
            # Compute weighted total score for each LLM
            df['LLM_1_Total'] = df['LLM_1_Classification'].apply(
                lambda x: (
                    (x.get('Citation', 0) * 0.25) +
                    (x.get('Recency', 0) * 0.20) +
                    (x.get('TopicAlignment', 0) * 0.15) +
                    (x.get('Clarity', 0) * 0.13) +
                    (x.get('Comprehensiveness', 0) * 0.10)
                )
            )
            df['LLM_2_Total'] = df['LLM_2_Classification'].apply(
                lambda x: (
                    (x.get('Citation', 0) * 0.25) +
                    (x.get('Recency', 0) * 0.20) +
                    (x.get('TopicAlignment', 0) * 0.15) +
                    (x.get('Clarity', 0) * 0.13) +
                    (x.get('Comprehensiveness', 0) * 0.10)
                )
            )
            df['LLM_3_Total'] = df['LLM_3_Classification'].apply(
                lambda x: (
                    (x.get('Citation', 0) * 0.25) +
                    (x.get('Recency', 0) * 0.20) +
                    (x.get('TopicAlignment', 0) * 0.15) +
                    (x.get('Clarity', 0) * 0.13) +
                    (x.get('Comprehensiveness', 0) * 0.10)
                )
            )
            df['LLM_4_Total'] = df['LLM_4_Classification'].apply(
                lambda x: (
                    (x.get('Citation', 0) * 0.25) +
                    (x.get('Recency', 0) * 0.20) +
                    (x.get('TopicAlignment', 0) * 0.15) +
                    (x.get('Clarity', 0) * 0.13) +
                    (x.get('Comprehensiveness', 0) * 0.10)
                )
            )
            
            # Save to new CSV
            df.to_csv(output_file, index=False)
            
            logger.info(f"Classification completed. Results saved to {output_file}")
        
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
        except PermissionError:
            logger.error(f"Permission denied when writing to {output_file}")
        except Exception as e:
            logger.error(f"Unexpected error processing CSV: {e}")

def main():
    """Main execution function"""
    input_file = os.path.expanduser('~/Documents/Abhave/CodingProjects/AceScholarsPrognostic/aidiagnostics/response.csv')
    output_file = os.path.expanduser('~/Documents/Abhave/CodingProjects/AceScholarsPrognostic/aidiagnostics/classified_responses.csv')
    
    try:
        classifier = ResponseClassifier()
        classifier.process_csv(input_file, output_file)
    
    except Exception as e:
        logger.error(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
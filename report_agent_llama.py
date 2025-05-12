import pandas as pd
import numpy as np
import time
from datetime import datetime
import re
import os
import json
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InterviewAnalyzer:
    """
    A class that analyzes interview transcripts and generates a comprehensive report
    about the candidate using Llama-2-7b-chat-hf model.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: str = "cuda"):
        """
        Initialize the InterviewAnalyzer with a Hugging Face model.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        
        # Check if CUDA is available when device is set to 'cuda'
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Initialize tokenizer and model
        logger.info(f"Loading tokenizer and model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Enable token merging for faster inference
        torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        
        # Load model with appropriate configuration for speed
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            low_cpu_mem_usage=True
        )
        
        # Create a text generation pipeline (no `device` argument)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        logger.info(f"Initialized InterviewAnalyzer with model: {model_name} on {self.device}")
    
    def load_transcript(self, file_path: str) -> Tuple[pd.DataFrame, float]:
        """
        Load the transcript from a CSV file.
        
        Args:
            file_path: Path to the transcript CSV file
            
        Returns:
            Tuple of DataFrame containing the transcript and the duration of the interview
        """
        try:
            records = []
            with open(file_path, encoding='utf-8') as f:
                # Saltar la cabecera
                next(f)
                for line in f:
                    line = line.rstrip('\n')
                    if not line:
                        continue
                    # Partir sólo por la primera coma
                    timestamp_str, text = line.split(',', 1)
                    try:
                        ts = float(timestamp_str)
                    except ValueError:
                        logger.warning(f"Ignoring invalid timestamp: {timestamp_str}")
                        continue
                    records.append({'timestamp_s': ts, 'text': text})
            
            df = pd.DataFrame(records)
            interview_duration = df['timestamp_s'].max()
            logger.info(f"Loaded transcript with {len(df)} entries, duration: {interview_duration:.2f}s")
            return df, interview_duration

        except Exception as e:
            logger.error(f"Error loading transcript: {str(e)}")
            raise
    
    def extract_candidate_info(self, transcript_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract key information about the candidate from the transcript using Llama model.
        
        Args:
            transcript_df: DataFrame containing the transcript
            
        Returns:
            Dictionary containing extracted candidate information
        """
        # Combine all transcript text for processing
        full_text = ' '.join(transcript_df['text'].tolist())
        
        # Limit text length to avoid token limits
        if len(full_text) > 6000:
            full_text = full_text[:6000] + "..."
        
        # Create a structured prompt to extract key information in Llama-2 chat format
        # Llama-2 models use a specific chat format with B_INST and E_INST tokens
        extraction_prompt = f"""<s>[INST] You are an AI assistant that extracts key information from interview transcripts and formats it as structured JSON.

Extract the following information about the job candidate from this interview transcript.
If any information is not found, respond with "Not mentioned".

Interview transcript:
{full_text}

Extract:
1. Full name of the candidate
2. Current job position and previous professional experience
3. Academic background (university and degree)
4. Skills mentioned by the candidate

Format your response as JSON with these keys: "name", "professional_experience", "academic_background", "skills". 
Only return valid JSON, no other text. [/INST]"""
        
        logger.info("Extracting candidate information...")
        start_time = time.time()
        
        try:
            # Generate the response
            outputs = self.generator(
                extraction_prompt,
                max_new_tokens=500,
                do_sample=False,  # Use greedy decoding for information extraction
                temperature=0.1,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            response_text = outputs[0]['generated_text']
            
            # Extract only the JSON part from the response
            # Find JSON content between curly braces
            json_pattern = r'\{.*\}'
            json_matches = re.search(json_pattern, response_text, re.DOTALL)
            
            if json_matches:
                json_str = json_matches.group(0)
                # Parse the JSON response
                try:
                    info = json.loads(json_str)
                    # Ensure all required keys are present
                    required_keys = ["name", "professional_experience", "academic_background", "skills"]
                    for key in required_keys:
                        if key not in info:
                            info[key] = "Not mentioned"
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON, using default values")
                    info = {
                        "name": "Not mentioned",
                        "professional_experience": "Not mentioned",
                        "academic_background": "Not mentioned",
                        "skills": "Not mentioned"
                    }
            else:
                logger.warning("No JSON found in response, using default values")
                info = {
                    "name": "Not mentioned",
                    "professional_experience": "Not mentioned",
                    "academic_background": "Not mentioned",
                    "skills": "Not mentioned"
                }
            
            process_time = time.time() - start_time
            logger.info(f"Extracted candidate information in {process_time:.2f} seconds")
            
            return info
        except Exception as e:
            logger.error(f"Error extracting candidate information: {str(e)}")
            return {
                "name": "Not mentioned",
                "professional_experience": "Not mentioned",
                "academic_background": "Not mentioned",
                "skills": "Not mentioned"
            }
    
    def generate_summary(self, transcript_df: pd.DataFrame, candidate_info: Dict[str, Any]) -> str:
        """
        Generate a comprehensive summary of the candidate based on the transcript and
        extracted information using Llama model.
        
        Args:
            transcript_df: DataFrame containing the transcript
            candidate_info: Dictionary containing extracted candidate information
            
        Returns:
            Summary of the candidate as a string
        """
        # Limit the transcript for processing to avoid token limits
        texts = transcript_df['text'].tolist()
        
        if len(texts) > 30:
            first_third = texts[:len(texts)//3]
            last_third = texts[-len(texts)//3:]
            selected_texts = first_third + last_third
            full_text = ' '.join(selected_texts)
        else:
            full_text = ' '.join(texts)
            
        # Further limit text length if needed
        if len(full_text) > 4000:
            full_text = full_text[:4000] + "..."
        
        # Create a structured prompt for the summary in Llama-2 chat format
        summary_prompt = f"""<s>[INST] You are an expert HR professional who writes concise, insightful summaries of job candidates based on interview transcripts.

Generate a professional, 200-word summary of the job candidate based on this interview transcript and the extracted information.

Extracted candidate information:
- Name: {candidate_info['name']}
- Professional Experience: {candidate_info['professional_experience']}
- Academic Background: {candidate_info['academic_background']}
- Skills: {candidate_info['skills']}

Interview transcript excerpt:
{full_text}

Create a comprehensive professional summary that highlights the candidate's background, experience, skills, and suitability for a professional role.
The summary should be approximately 200 words, well-structured, and focus on the most relevant information for a hiring manager. [/INST]"""
        
        logger.info("Generating candidate summary...")
        start_time = time.time()
        
        try:
            # Generate the response
            outputs = self.generator(
                summary_prompt,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.7,  # Allow some creativity in the summary
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            full_response = outputs[0]['generated_text']
            
            # Remove the input prompt from the output
            summary = full_response.replace(summary_prompt, "").strip()
            
            # Clean up any extra formatting or instructions
            summary = re.sub(r'</?s>|\[/?INST\]', '', summary).strip()
            
            process_time = time.time() - start_time
            logger.info(f"Generated summary in {process_time:.2f} seconds")
            
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate summary due to an error."
    
    def format_report(self, candidate_info: Dict[str, Any], summary: str, interview_duration: float) -> str:
        """
        Format the final report according to the required structure.
        
        Args:
            candidate_info: Dictionary containing extracted candidate information
            summary: Generated summary of the candidate
            interview_duration: Duration of the interview in seconds
            
        Returns:
            Formatted report as a string
        """
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format duration in minutes and seconds
        minutes = int(interview_duration // 60)
        seconds = int(interview_duration % 60)
        duration_str = f"{minutes} minutes and {seconds} seconds"
        
        report = f"""Name: {candidate_info['name']}
Date: {current_datetime}
Professional Experience: {candidate_info['professional_experience']}
Academic Background: {candidate_info['academic_background']}
Skills: {candidate_info['skills']}
Time: {duration_str}

SUMMARY:
{summary}
"""
        return report
    
    def analyze_interview(self, transcript_path: str) -> str:
        """
        Main method that orchestrates the entire analysis process.
        
        Args:
            transcript_path: Path to the transcript CSV file
            
        Returns:
            Formatted report as a string
        """
        logger.info(f"Starting interview analysis for transcript: {transcript_path}")
        overall_start_time = time.time()
        
        # Load transcript
        transcript_df, interview_duration = self.load_transcript(transcript_path)
        
        # Extract candidate information
        candidate_info = self.extract_candidate_info(transcript_df)
        
        # Generate summary
        summary = self.generate_summary(transcript_df, candidate_info)
        
        # Format the final report
        report = self.format_report(candidate_info, summary, interview_duration)
        
        overall_process_time = time.time() - overall_start_time
        logger.info(f"Completed interview analysis in {overall_process_time:.2f} seconds")
        
        return report

def main():
    """
    Main function to run the interview analyzer.
    """
    # Path to transcript file
    transcript_path = "data/transcripts.csv"
    
    # Check if the transcript file exists
    if not os.path.exists(transcript_path):
        print(f"Error: Transcript file '{transcript_path}' not found")
        return
    
    # Determine device (CUDA or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not available, using CPU (this will be slower)")
    
    # Initialize and run the analyzer
    print("Loading Llama-2-7b-chat-hf model...")
    analyzer = InterviewAnalyzer(model_name="meta-llama/Llama-2-7b-chat-hf", device=device)
    
    print("Analyzing interview transcript...")
    start_time = time.time()
    
    # Generate report
    report = analyzer.analyze_interview(transcript_path)
    
    # Output report
    print("\n" + "="*50)
    print("INTERVIEW ANALYSIS REPORT")
    print("="*50)
    print(report)
    print("="*50)
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save report to fixed file path in data directory
    report_filename = os.path.join(data_dir, "report_llama.txt")
    with open(report_filename, "w") as f:
        f.write(report)
    print(f"Report saved to {report_filename}")
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

# Add function to optimize Llama model for speed
def optimize_llama_for_speed():
    """
    Apply various optimizations to speed up Llama model inference on GPU.
    """
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set appropriate PyTorch optimizations
        torch.set_float32_matmul_precision('high')
        
        logger.info("Applied CUDA and PyTorch optimizations for faster inference")

if __name__ == "__main__":
    # Apply optimizations before running
    optimize_llama_for_speed()
    main()
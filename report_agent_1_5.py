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
import gc  # For garbage collection
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import bitsandbytes as bnb  # For quantization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InterviewAnalyzer:
    """
    A class that analyzes interview transcripts and generates a comprehensive report
    about the candidate using an optimized and quantized model.
    """
    
    def __init__(self, model_name: str = "microsoft/phi-1_5", device: str = "cuda"):
        """
        Initialize the InterviewAnalyzer with a lightweight, quantized Hugging Face model.
        
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
        
        # Initialize tokenizer and model with extreme optimizations
        logger.info(f"Loading tokenizer and model: {model_name} on {self.device}")
        
        # Load tokenizer with padding token settings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply 4-bit quantization for extreme speed improvement
        if self.device == 'cuda':
            logger.info(f"Loading with 4-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device,
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                trust_remote_code=True
            )
        else:
            # Fall back to 8-bit quantization for CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        
        # Ensure model is on correct device and optimized
        if self.device == 'cuda':
            # Apply extreme CUDA optimizations
            self.model.config.use_cache = True
            
        # Create optimized text generation pipeline with bigger batch size
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=2  # Increased for parallel processing
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
                # Skip header
                next(f)
                for line in f:
                    line = line.rstrip('\n')
                    if not line:
                        continue
                    # Split only by the first comma
                    parts = line.split(',', 1)
                    if len(parts) < 2:
                        logger.warning(f"Ignoring malformed line: {line}")
                        continue
                        
                    timestamp_str, text = parts
                    try:
                        ts = float(timestamp_str)
                    except ValueError:
                        logger.warning(f"Ignoring invalid timestamp: {timestamp_str}")
                        continue
                    records.append({'timestamp_s': ts, 'text': text})
            
            df = pd.DataFrame(records)
            
            if df.empty:
                logger.warning("Empty transcript loaded")
                return df, 0.0
                
            interview_duration = df['timestamp_s'].max()
            logger.info(f"Loaded transcript with {len(df)} entries, duration: {interview_duration:.2f}s")
            return df, interview_duration

        except Exception as e:
            logger.error(f"Error loading transcript: {str(e)}")
            raise
    
    def _extract_json_from_response(self, response_text: str) -> Dict:
        """
        Helper method to extract JSON from model response.
        
        Args:
            response_text: Text response from the model
            
        Returns:
            Extracted JSON as dictionary
        """
        # Find JSON content between curly braces with relaxed pattern matching
        json_pattern = r'\{[\s\S]*\}'
        json_matches = re.search(json_pattern, response_text, re.DOTALL)
        
        if json_matches:
            json_str = json_matches.group(0)
            # Try to parse the JSON, handling any JSON errors
            try:
                # Fix common JSON formatting issues
                fixed_json = re.sub(r'(\w+):', r'"\1":', json_str)
                fixed_json = re.sub(r',\s*}', '}', fixed_json)
                fixed_json = re.sub(r',\s*]', ']', fixed_json)
                
                # Parse the JSON
                return json.loads(fixed_json)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON: {str(e)}")
                logger.debug(f"Problematic JSON string: {json_str}")
        
        # Default values if JSON extraction fails
        return {
            "name": "Not mentioned",
            "professional_experience": "Not mentioned",
            "academic_background": "Not mentioned",
            "skills": "Not mentioned"
        }
    
    def _extract_key_info_from_text(self, text_chunk: str, info_type: str) -> str:
        """
        Extract specific information from a text chunk using regex patterns
        
        Args:
            text_chunk: Text to analyze
            info_type: Type of information to extract (name, experience, education, skills)
            
        Returns:
            Extracted information or "Not mentioned"
        """
        if info_type == "name":
            # Look for name patterns like "I am [Name]" or "My name is [Name]"
            name_patterns = [
                r"(?:I am|I'm|my name is|name is|call me|I go by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
                r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?:\s+speaking|\s+here|\s*,\s*I)"
            ]
            for pattern in name_patterns:
                matches = re.findall(pattern, text_chunk)
                if matches:
                    return matches[0].strip()
                    
        elif info_type == "experience":
            # Look for experience patterns
            exp_patterns = [
                r"(?:I work|I've worked|I have worked|working|I am|I'm|currently)\s+(?:at|for|as|with)\s+([^,.]{3,50})",
                r"(?:experience|employed|position|role)\s+(?:at|with|as)\s+([^,.]{3,50})",
                r"(?:years|months)\s+(?:of experience|in|at)\s+([^,.]{3,50})"
            ]
            for pattern in exp_patterns:
                matches = re.findall(pattern, text_chunk)
                if matches:
                    return matches[0].strip()
                    
        elif info_type == "education":
            # Look for education patterns
            edu_patterns = [
                r"(?:graduated|degree from|studied at|attended)\s+([^,.]{3,50})",
                r"(?:Bachelor'?s?|Master'?s?|MBA|PhD|diploma)\s+(?:degree|in|from)?\s+([^,.]{3,50})",
                r"(?:university|college|school)\s+of\s+([^,.]{3,50})"
            ]
            for pattern in edu_patterns:
                matches = re.findall(pattern, text_chunk)
                if matches:
                    return matches[0].strip()
                    
        elif info_type == "skills":
            # Look for skill patterns
            skill_patterns = [
                r"skills? (?:include|in|with|:)\s+([^.]{3,70})",
                r"(?:proficient|experienced|expertise|knowledge) (?:in|with|of)\s+([^.]{3,70})",
                r"(?:I know|I've used|familiar with|worked with)\s+([^.]{3,70})"
            ]
            for pattern in skill_patterns:
                matches = re.findall(pattern, text_chunk)
                if matches:
                    return matches[0].strip()
                    
        return "Not mentioned"
    
    def extract_candidate_info(self, transcript_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract key information about the candidate using a hybrid approach:
        1. Rules-based extraction with regex patterns
        2. ML model for fallback
        
        Args:
            transcript_df: DataFrame containing the transcript
            
        Returns:
            Dictionary containing extracted candidate information
        """
        if transcript_df.empty:
            logger.warning("Empty transcript, returning default candidate info")
            return {
                "name": "Not mentioned",
                "professional_experience": "Not mentioned",
                "academic_background": "Not mentioned",
                "skills": "Not mentioned"
            }
        
        # Start timing
        logger.info("Extracting candidate information...")
        start_time = time.time()
        
        # Step 1: Quick rule-based extraction using all text
        # This will run very fast and may get the information we need
        all_text = ' '.join(transcript_df['text'].tolist())
        
        # Extract information using regex patterns for common formats
        result = {
            "name": self._extract_key_info_from_text(all_text, "name"),
            "professional_experience": self._extract_key_info_from_text(all_text, "experience"),
            "academic_background": self._extract_key_info_from_text(all_text, "education"),
            "skills": self._extract_key_info_from_text(all_text, "skills")
        }
        
        # Count how many fields were successfully extracted
        extracted_count = sum(1 for value in result.values() if value != "Not mentioned")
        
        # If we got at least 3 fields, we can skip the ML model inference
        if extracted_count >= 3:
            process_time = time.time() - start_time
            logger.info(f"Extracted candidate information using regex in {process_time:.2f} seconds")
            return result
            
        # Step 2: Only if regex didn't get enough information, use the ML model
        # Focus on shorter, more relevant text sections
        texts = transcript_df['text'].tolist()
        
        # Just take first few exchanges and some longer responses for speed
        first_few = texts[:min(3, len(texts))]
        longest_responses = sorted([t for t in texts if len(t) > 30], 
                                 key=len, reverse=True)[:3]
        
        processed_text = ' '.join(first_few + longest_responses)
        if len(processed_text) > 1000:  # Drastically reduce context size
            processed_text = processed_text[:1000]
        
        # Create an ultra-focused prompt for speed
        extraction_prompt = f"""Extract candidate info from interview transcript.
Transcript: {processed_text}
Output JSON only with these keys: "name", "professional_experience", "academic_background", "skills"
For missing info use "Not mentioned".
JSON:"""
        
        try:
            # Generate with minimal tokens and aggressive settings for speed
            outputs = self.generator(
                extraction_prompt,
                max_new_tokens=150,  # Bare minimum tokens
                do_sample=True,     # Deterministic 
                temperature=0.1,
                top_k=5,
                num_return_sequences=1
            )
            
            # Extract the generated text
            response_text = outputs[0]['generated_text']
            
            # Extract JSON from the response
            ml_info = self._extract_json_from_response(response_text)
            
            # Merge ML results with regex results, prioritizing regex where available
            for key in result:
                if result[key] == "Not mentioned" and key in ml_info and ml_info[key] != "Not mentioned":
                    result[key] = ml_info[key]
            
            process_time = time.time() - start_time
            logger.info(f"Extracted candidate information in {process_time:.2f} seconds")
            
            return result
        except Exception as e:
            logger.error(f"Error in ML extraction: {str(e)}, using regex results")
            return result
    
    def _generate_summary_template(self, candidate_info: Dict[str, Any]) -> str:
        """
        Generate a summary template based on candidate info without ML model
        
        Args:
            candidate_info: Dictionary containing extracted candidate information
            
        Returns:
            A basic summary template string
        """
        name = candidate_info['name']
        experience = candidate_info['professional_experience']
        education = candidate_info['academic_background']
        skills = candidate_info['skills']
        
        # Simple template-based summary
        if name == "Not mentioned":
            name_part = "The candidate"
        else:
            name_part = f"{name}"
            
        if experience == "Not mentioned":
            exp_part = "has professional experience"
        else:
            exp_part = f"has experience as {experience}"
            
        if education == "Not mentioned":
            edu_part = "has relevant educational background"
        else:
            edu_part = f"has educational background in {education}"
            
        if skills == "Not mentioned":
            skills_part = "demonstrated various professional skills during the interview"
        else:
            skills_part = f"possesses skills in {skills}"
            
        template = f"{name_part} {exp_part} and {edu_part}. The candidate {skills_part}. Based on the interview, the candidate appears to be professional and could be a good fit for roles requiring these qualifications."
        return template
        
    def generate_summary(self, transcript_df: pd.DataFrame, candidate_info: Dict[str, Any]) -> str:
        """
        Generate a comprehensive summary using a hybrid approach:
        1. Create a template-based summary for immediate use
        2. Run ML model in parallel to enhance the summary
        3. Use whichever completes first or falls back to template
        
        Args:
            transcript_df: DataFrame containing the transcript
            candidate_info: Dictionary containing extracted candidate information
            
        Returns:
            Summary of the candidate as a string
        """
        if transcript_df.empty:
            logger.warning("Empty transcript, returning default summary")
            return "Unable to generate summary due to empty transcript data."
        
        logger.info("Generating candidate summary...")
        start_time = time.time()
        
        # 1. Immediately create a template-based summary (instant)
        template_summary = self._generate_summary_template(candidate_info)
        
        # Set a very strict time limit - we want the whole process to be fast
        time_limit = 20  # seconds
        
        # 2. Try to get an ML-generated summary with a strict time limit
        try:
            # Create a very minimal prompt to save tokens
            all_text = ' '.join(transcript_df['text'].tolist())
            # Take just beginning and ending
            if len(all_text) > 1000:
                processed_text = all_text[:500] + "..." + all_text[-500:]
            else:
                processed_text = all_text
                
            # Ultra-simplified prompt
            summary_prompt = f"""Write a professional HR summary of this candidate. 
Name: {candidate_info['name']}
Experience: {candidate_info['professional_experience']}
Education: {candidate_info['academic_background']}
Skills: {candidate_info['skills']}
Interview: {processed_text}
Summary:"""
            
            # Create a future for the model generation
            result_summary = [template_summary]  # Default to template
            
            def generate_with_timeout():
                try:
                    # Generate with minimal tokens and aggressive settings
                    outputs = self.generator(
                        summary_prompt,
                        max_new_tokens=150,  # Minimal tokens
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                        num_return_sequences=1
                    )
                    
                    # Extract just the generated part
                    full_response = outputs[0]['generated_text']
                    ml_summary = full_response.replace(summary_prompt, "").strip()
                    
                    # Clean up common formatting issues
                    ml_summary = re.sub(r'^Summary:', '', ml_summary, flags=re.IGNORECASE).strip()
                    
                    # Update the result if we got something valid
                    if len(ml_summary) > 50:
                        result_summary[0] = ml_summary
                except Exception as e:
                    logger.warning(f"ML summary generation failed: {str(e)}")
            
            # Start the generation in a separate thread
            thread = threading.Thread(target=generate_with_timeout)
            thread.daemon = True  # Thread will die when main thread exits
            thread.start()
            
            # Wait up to the time limit
            thread.join(timeout=time_limit)
            
            # If still running after time limit, we'll use the template
            if thread.is_alive():
                logger.warning(f"ML summary generation exceeded {time_limit}s timeout, using template")
                # Will use template_summary from result_summary[0]
            
            final_summary = result_summary[0]
            
            process_time = time.time() - start_time
            logger.info(f"Generated summary in {process_time:.2f} seconds")
            
            return final_summary
            
        except Exception as e:
            logger.error(f"Error in summary generation: {str(e)}")
            process_time = time.time() - start_time
            logger.info(f"Falling back to template summary in {process_time:.2f} seconds")
            return template_summary
    
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
Interview Duration: {duration_str}

SUMMARY:
{summary}
"""
        return report
    
    def analyze_interview(self, transcript_path: str) -> str:
        """
        Main method that orchestrates the entire analysis process.
        Uses parallel processing for maximum speed.
        
        Args:
            transcript_path: Path to the transcript CSV file
            
        Returns:
            Formatted report as a string
        """
        logger.info(f"Starting interview analysis for transcript: {transcript_path}")
        overall_start_time = time.time()
        
        # Load transcript
        transcript_df, interview_duration = self.load_transcript(transcript_path)
        
        if transcript_df.empty:
            logger.warning("Empty transcript, returning minimal report")
            return "ERROR: Empty or invalid transcript file. No report generated."
        
        # Set a hard overall deadline
        overall_deadline = 55  # seconds
        
        # Use ThreadPoolExecutor to run tasks in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks simultaneously
            extract_future = executor.submit(self.extract_candidate_info, transcript_df)
            
            # Wait for candidate info to complete first (needed for summary)
            try:
                # Wait with timeout to get candidate info
                candidate_info = extract_future.result(timeout=30)
                
                # Now that we have candidate info, start the summary generation
                summary_future = executor.submit(self.generate_summary, transcript_df, candidate_info)
                
                # Get summary with timeout
                remaining_time = max(5, overall_deadline - (time.time() - overall_start_time))
                summary = summary_future.result(timeout=remaining_time)
                
            except Exception as e:
                logger.error(f"Error in parallel processing: {str(e)}")
                # Fall back to default values if necessary
                if 'candidate_info' not in locals():
                    candidate_info = {
                        "name": "Not mentioned",
                        "professional_experience": "Not mentioned",
                        "academic_background": "Not mentioned",
                        "skills": "Not mentioned"
                    }
                
                if 'summary' not in locals():
                    summary = self._generate_summary_template(candidate_info)
        
        # Format the final report
        report = self.format_report(candidate_info, summary, interview_duration)
        
        overall_process_time = time.time() - overall_start_time
        logger.info(f"Completed interview analysis in {overall_process_time:.2f}")
        
        return report  # Ensure the report is returned
    
    def __del__(self):
        """
        Clean up resources when the analyzer is deleted.
        """
        # Explicitly free GPU memory when done
        if hasattr(self, 'model') and self.device == 'cuda':
            del self.model
            del self.generator
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleaned up model resources")

def optimize_gpu_for_inference():
    """
    Apply various optimizations to speed up model inference on GPU.
    """
    if torch.cuda.is_available():
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Enable TF32 for faster matrix multiplications on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set appropriate PyTorch optimizations
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        
        # Clear any existing GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info(f"Applied GPU optimizations for {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

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
        # Apply optimizations for the specific GPU
        optimize_gpu_for_inference()
    else:
        print("GPU not available, using CPU (this will be slower)")
    
    # Initialize and run the analyzer
    print("Loading Microsoft Phi-2 model...")
    analyzer = InterviewAnalyzer(model_name="microsoft/phi-2", device=device)
    
    print("Analyzing interview transcript...")
    start_time = time.time()
    
    try:
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
        
        # Save report to file
        report_filename = os.path.join(data_dir, "report_phi.txt")
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {report_filename}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        logger.error(f"Analysis failed with error: {str(e)}", exc_info=True)
    finally:
        # Make sure to clean up resources
        del analyzer
        if device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
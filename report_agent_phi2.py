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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InterviewAnalyzer:
    """
    A class that analyzes interview transcripts and generates a comprehensive report
    about the candidate using Microsoft's Phi-2 model.
    """
    
    def __init__(self, model_name: str = "microsoft/phi-2", device: str = "cuda"):
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
        
        # Initialize tokenizer and model with optimizations
        logger.info(f"Loading tokenizer and model: {model_name} on {self.device}")
        
        # Load tokenizer with padding token settings
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Enable token merging for faster inference
        torch_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        
        # Load model with optimizations for speed
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Apply model optimizations
        if self.device == 'cuda':
            self.model = self.model.to('cuda')
            # Apply Flash Attention if available
            # (Phi-2 doesn't natively support this, but we've added the setting anyway)
            self.model.config.use_flash_attention_2 = True
            
        # Create optimized text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=1  # Process one input at a time for reliability
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
    
    def extract_candidate_info(self, transcript_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract key information about the candidate from the transcript using Phi-2 model.
        Optimized for speed with focused prompt and processing.
        
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
        
        # Process the transcript text to extract relevant sections
        # Focus on the beginning for intro/background and select key parts
        texts = transcript_df['text'].tolist()
        
        # Create a focused subset of the text to optimize processing
        # Take first 25% (introductions) and sample from rest
        first_quarter = texts[:max(3, len(texts)//4)]
        
        # Take samples from the rest, focusing on longer responses (likely candidate answers)
        rest_of_text = texts[len(first_quarter):]
        
        # Select longer texts that are more likely to contain info
        informative_texts = [t for t in rest_of_text if len(t) > 50]
        if len(informative_texts) > 10:
            # Sample from longer texts for processing efficiency
            sampled_texts = np.random.choice(informative_texts, min(10, len(informative_texts)), replace=False).tolist()
        else:
            sampled_texts = informative_texts
        
        # Combine texts for processing
        processed_text = ' '.join(first_quarter + sampled_texts)
        
        # Limit text length to avoid token limits
        if len(processed_text) > 2000:  # Smaller context for faster processing
            processed_text = processed_text[:2000] + "..."
        
        # Create a focused prompt for Phi-2
        extraction_prompt = f"""Task: Extract candidate information from an interview transcript.

Transcript:
{processed_text}

Instructions:
1. Extract the following information about the job candidate:
   - Full name
   - Current job position and/or professional experience
   - Academic background (university and degree)
   - Technical and soft skills mentioned

2. Format your response as JSON with these keys: "name", "professional_experience", "academic_background", "skills"
3. Use "Not mentioned" for any information not found in the transcript
4. Return only the JSON object, no other text

JSON Output:
"""
        
        logger.info("Extracting candidate information...")
        start_time = time.time()
        
        try:
            # Generate the response with optimized settings for information extraction
            outputs = self.generator(
                extraction_prompt,
                max_new_tokens=300,  # Reduced token count for faster processing
                do_sample=True,     # Deterministic output for information extraction
                temperature=0.3,     # Low temperature for factual responses
                top_k=10,            # Limited sampling for faster processing
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            response_text = outputs[0]['generated_text']
            
            # Extract JSON from the response
            info = self._extract_json_from_response(response_text)
            
            # Ensure all required keys are present
            required_keys = ["name", "professional_experience", "academic_background", "skills"]
            for key in required_keys:
                if key not in info:
                    info[key] = "Not mentioned"
            
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
        extracted information using Phi-2 model. Optimized for speed.
        
        Args:
            transcript_df: DataFrame containing the transcript
            candidate_info: Dictionary containing extracted candidate information
            
        Returns:
            Summary of the candidate as a string
        """
        if transcript_df.empty:
            logger.warning("Empty transcript, returning default summary")
            return "Unable to generate summary due to empty transcript data."
        
        # Intelligently select important parts of the transcript
        texts = transcript_df['text'].tolist()
        
        # Improved text selection strategy:
        # 1. Include introduction (first few exchanges)
        first_few = texts[:min(5, len(texts))]
        
        # 2. Include longer responses (likely substantive answers)
        longer_responses = [t for t in texts if len(t) > 80]
        if len(longer_responses) > 5:
            # Sample from longer responses
            selected_responses = np.random.choice(longer_responses, 5, replace=False).tolist()
        else:
            selected_responses = longer_responses
            
        # 3. Include conclusion (last few exchanges)
        last_few = texts[-min(3, len(texts)):]
        
        # Combine selected text parts
        selected_texts = first_few + selected_responses + last_few
        full_text = ' '.join(selected_texts)
            
        # Further limit text length for faster processing
        if len(full_text) > 1500:  # Reduced context size for faster processing
            full_text = full_text[:1500] + "..."
        
        # Create an optimized prompt for Phi-2
        summary_prompt = f"""Task: Write a professional summary of a job candidate based on interview data.

Candidate Information:
- Name: {candidate_info['name']}
- Professional Experience: {candidate_info['professional_experience']}
- Academic Background: {candidate_info['academic_background']}
- Skills: {candidate_info['skills']}

Interview Excerpts:
{full_text}

Instructions:
1. Write a concise, professional summary (about 150 words) highlighting the candidate's:
   - Background and experience
   - Key skills and qualifications
   - Potential fit for professional roles
2. Focus on factual information from the transcript
3. Use a professional, HR-friendly tone
4. Format as a single cohesive paragraph

Summary:
"""
        
        logger.info("Generating candidate summary...")
        start_time = time.time()
        
        try:
            # Generate the summary with optimized settings
            outputs = self.generator(
                summary_prompt,
                max_new_tokens=300,  # Sufficient for a concise summary
                do_sample=True,      # Allow some variation in the summary
                temperature=0.5,     # Balanced between creativity and factuality
                top_p=0.92,          # Slightly conservative sampling
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            full_response = outputs[0]['generated_text']
            
            # Remove the input prompt from the output to get just the summary
            summary = full_response.replace(summary_prompt, "").strip()
            
            # Clean up common formatting issues
            summary = re.sub(r'^Summary:', '', summary, flags=re.IGNORECASE).strip()
            
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
Interview Duration: {duration_str}

SUMMARY:
{summary}
"""
        return report
    
    def analyze_interview(self, transcript_path: str) -> str:
        """
        Main method that orchestrates the entire analysis process.
        Optimized for speed and efficiency.
        
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
        
        # Extract candidate information
        candidate_info = self.extract_candidate_info(transcript_df)
        
        # Generate summary
        summary = self.generate_summary(transcript_df, candidate_info)
        
        # Format the final report
        report = self.format_report(candidate_info, summary, interview_duration)
        
        overall_process_time = time.time() - overall_start_time
        logger.info(f"Completed interview analysis in {overall_process_time:.2f} seconds")
        
        return report
    
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
        report_filename = os.path.join(data_dir, "report_phi2.txt")
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
"""
report_agent_improved.py — Advanced interview‑report generator with instruction-tuned model
=================================================================
• Lee data/transcripts.csv y crea un informe profesional detallado en Markdown.
• Analiza y extrae información relevante del candidato: nombre, experiencia, educación, habilidades.
• Usa técnicas avanzadas de procesamiento de texto para mejorar la extracción de información.
• Optimizado para NVIDIA RTX 500 Ada Generation Laptop GPU

Requisitos:
  pip install -U transformers accelerate sentencepiece optimum peft spacy
"""

from __future__ import annotations
import os, csv, re, warnings, time, json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import timedelta
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    pipeline, set_seed
)

# ------------------------------------------------------------------------- #
# Config
# ------------------------------------------------------------------------- #
TRANSCRIPT_CSV = Path("data/transcripts.csv")
REPORT_PATH    = Path("data/conversation_report.md")

# Modelo instruction-tuned para mejores resultados
MODEL_NAME     = "microsoft/Phi-3-mini-4k-instruct"  # Alternativa: "google/gemma-2b-it"
CTX_LIMIT      = 2500      # Ventana de contexto
MAX_NEW_TOK    = 800       # Más tokens para respuestas completas
SEED           = 42        # Para reproducibilidad

# ------------------------------------------------------------------------- #
# Helper functions - Transcript Processing
# ------------------------------------------------------------------------- #
def read_transcript(p: Path) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Lee el CSV y extrae datos estructurados y el texto de la transcripción"""
    if not p.exists():
        raise FileNotFoundError("Run AudioTranscriber first → data/transcripts.csv")
    
    metadata = {
        "candidate_name": "Unspecified",
        "roles": set(),
        "interview_duration": 0,
        "total_exchanges": 0
    }
    
    rows = []
    with p.open(encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        
        # Calcular duración de la entrevista
        if rows:
            try:
                first_timestamp = float(rows[0]['timestamp_s'])
                last_timestamp = float(rows[-1]['timestamp_s'])
                metadata["interview_duration"] = last_timestamp - first_timestamp
            except (ValueError, KeyError):
                metadata["interview_duration"] = 0
        
        metadata["total_exchanges"] = len(rows)
    
    return rows, metadata

def extract_candidate_info(transcript_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Extrae información detallada del candidato de la transcripción"""
    info = {
        "name": "Unspecified",
        "professional_experience": [],
        "academic_background": [],
        "strengths": [],
        "skills": [],
        "teamwork": [],
        "position_applied": "Unspecified"
    }
    
    # Patrones para detectar información específica
    name_patterns = [
        r"(?:my name is|I am|I'm) ([A-Z][a-z]+(?: [A-Z][a-z]+)+)",
        r"([A-Z][a-z]+(?: [A-Z][a-z]+)+) (?:speaking|here)"
    ]
    
    experience_patterns = [
        r"(?:worked|working) (?:at|for|with) ([^.,]+)",
        r"(?:I have|I've)(?: been)? (?:a|an) ([^.,]+ (?:for|since) [^.,]+)",
        r"(?:I am|I'm)(?: currently)? (?:a|an) ([^.,]+)",
        r"my (?:current|previous) (?:role|position|job) (?:is|was) ([^.,]+)",
        r"(?:I have|I've) ([0-9]+) years? (?:of)? experience (?:in|with) ([^.,]+)"
    ]
    
    education_patterns = [
        r"(?:I|my) (?:have|has|earned|studied|completed) (?:a|an) ([^.,]+ (?:degree|certificate|diploma)) (?:in|from) ([^.,]+)",
        r"(?:graduated|studied) (?:from|at) ([^.,]+) with (?:a|an) ([^.,]+)",
        r"(?:I have|I've got|I hold) (?:a|an) ([^.,]+ (?:in|from) [^.,]+)"
    ]
    
    skill_patterns = [
        r"(?:skilled|proficient|experienced|specialized) in ([^.,]+)",
        r"(?:my|I have) skills? (?:include|in|with) ([^.,]+)",
        r"I (?:can|know how to) ([^.,]+)",
        r"(?:expertise|knowledge|experience) (?:in|with) ([^.,]+)"
    ]
    
    strength_patterns = [
        r"(?:my|one of my) strength(?:s)? (?:is|are|include) ([^.,]+)",
        r"I (?:excel at|am good at|am strong in) ([^.,]+)",
        r"I consider myself (?:to be )?(?:a|an) ([^.,]+)"
    ]
    
    teamwork_patterns = [
        r"(?:worked|working|collaborate) (?:in|with) (?:a|the) team",
        r"team (?:player|work|collaboration|environment)",
        r"(?:I've|I have) led (?:a|the) team",
        r"responsible for (?:a|the) team",
        r"collaborative (?:environment|project|work)"
    ]
    
    position_patterns = [
        r"(?:applying|applied) for (?:the|a) ([^.,]+) position",
        r"(?:interested in|looking for) (?:the|a|an) ([^.,]+) role",
        r"position of ([^.,]+)",
        r"this ([^.,]+) opportunity"
    ]
    
    # Procesar cada fila de la transcripción
    for row in transcript_rows:
        text = row['text'].strip()
        
        # Detectar nombre del candidato
        if info["name"] == "Unspecified":
            for pattern in name_patterns:
                name_match = re.search(pattern, text, re.IGNORECASE)
                if name_match:
                    info["name"] = name_match.group(1)
                    break
        
        # Detectar experiencia profesional
        for pattern in experience_patterns:
            exp_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in exp_matches:
                if isinstance(match, tuple):
                    experience = " ".join(match)
                else:
                    experience = match
                if experience and experience not in info["professional_experience"]:
                    info["professional_experience"].append(experience)
        
        # Detectar formación académica
        for pattern in education_patterns:
            edu_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in edu_matches:
                if isinstance(match, tuple):
                    education = " ".join(match)
                else:
                    education = match
                if education and education not in info["academic_background"]:
                    info["academic_background"].append(education)
        
        # Detectar habilidades
        for pattern in skill_patterns:
            skill_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in skill_matches:
                skills = [s.strip() for s in match.split(',')]
                for skill in skills:
                    if skill and skill not in info["skills"] and len(skill) > 3:
                        info["skills"].append(skill)
        
        # Detectar fortalezas
        for pattern in strength_patterns:
            strength_matches = re.findall(pattern, text, re.IGNORECASE)
            for match in strength_matches:
                strengths = [s.strip() for s in match.split(',')]
                for strength in strengths:
                    if strength and strength not in info["strengths"] and len(strength) > 3:
                        info["strengths"].append(strength)
        
        # Detectar trabajo en equipo
        for pattern in teamwork_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                teamwork_context = text[:200]  # Tomar el contexto cercano
                if teamwork_context not in info["teamwork"]:
                    info["teamwork"].append(teamwork_context)
        
        # Detectar puesto solicitado
        if info["position_applied"] == "Unspecified":
            for pattern in position_patterns:
                position_match = re.search(pattern, text, re.IGNORECASE)
                if position_match:
                    info["position_applied"] = position_match.group(1)
                    break
    
    return info

def segment_transcript(transcript_rows: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """Segmenta la transcripción en secciones temáticas para mejor procesamiento"""
    segments = {
        "introduction": [],
        "experience": [],
        "education": [],
        "skills": [],
        "motivation": [],
        "questions": [],
        "conclusion": []
    }
    
    # Palabras clave para identificar segmentos
    keywords = {
        "introduction": ["introduce", "name", "background", "yourself", "myself"],
        "experience": ["experience", "work", "job", "career", "position", "role", "company"],
        "education": ["education", "degree", "university", "college", "school", "study", "graduate"],
        "skills": ["skill", "ability", "strength", "proficient", "knowledge", "expertise"],
        "motivation": ["motivation", "interest", "why", "goal", "aspiration", "future"],
        "questions": ["question", "ask", "wonder", "curious", "concern"]
    }
    
    # Identificar la sección para cada fila basado en palabras clave
    for row in transcript_rows:
        text = row['text'].lower()
        segment_assigned = False
        
        # Asignar a la introducción las primeras intervenciones
        if float(row['timestamp_s']) < 60 and not segments["introduction"]:
            segments["introduction"].append(row['text'])
            segment_assigned = True
            continue
            
        # Asignar a la conclusión las últimas intervenciones
        if float(row['timestamp_s']) > (float(transcript_rows[-1]['timestamp_s']) - 60) and not segment_assigned:
            segments["conclusion"].append(row['text'])
            segment_assigned = True
            continue
        
        # Asignar basado en palabras clave
        for segment, words in keywords.items():
            if any(keyword in text for keyword in words) and not segment_assigned:
                segments[segment].append(row['text'])
                segment_assigned = True
                break
        
        # Si no se asignó a ningún segmento, ponerlo en el segmento más probable
        if not segment_assigned:
            if "?" in text:
                segments["questions"].append(row['text'])
            else:
                max_score = 0
                best_segment = "experience"  # Default
                
                for segment, words in keywords.items():
                    score = sum(1 for word in words if word in text)
                    if score > max_score:
                        max_score = score
                        best_segment = segment
                
                segments[best_segment].append(row['text'])
    
    return segments

def prepare_transcript_context(transcript_rows: List[Dict[str, str]], 
                              candidate_info: Dict[str, Any],
                              segments: Dict[str, List[str]],
                              tokenizer, 
                              max_tokens: int) -> str:
    """Prepara un contexto optimizado de la transcripción para el modelo"""
    # Crear un contexto estructurado con las secciones más relevantes
    structured_context = []
    
    # Añadir la introducción
    if segments["introduction"]:
        structured_context.append("### INTRODUCTION")
        structured_context.extend(segments["introduction"][:3])  # Primeras 3 líneas
    
    # Añadir experiencia profesional
    if segments["experience"]:
        structured_context.append("### PROFESSIONAL EXPERIENCE")
        structured_context.extend(segments["experience"][:5])  # 5 menciones de experiencia
    
    # Añadir educación
    if segments["education"]:
        structured_context.append("### EDUCATION")
        structured_context.extend(segments["education"][:3])  # 3 menciones de educación
    
    # Añadir habilidades
    if segments["skills"]:
        structured_context.append("### SKILLS")
        structured_context.extend(segments["skills"][:5])  # 5 menciones de habilidades
    
    # Añadir motivación
    if segments["motivation"]:
        structured_context.append("### MOTIVATION")
        structured_context.extend(segments["motivation"][:3])  # 3 menciones de motivación
    
    # Añadir conclusión
    if segments["conclusion"]:
        structured_context.append("### CONCLUSION")
        structured_context.extend(segments["conclusion"][:2])  # 2 líneas de conclusión
    
    # Crear texto de contexto
    context_text = "\n\n".join(structured_context)
    
    # Añadir diálogo completo hasta el límite de tokens
    full_dialogue = "\n".join(f"[{row['timestamp_s']}s] {row['text']}" for row in transcript_rows)
    
    # Verificar si el contexto estructurado + diálogo completo cabe en el límite de tokens
    combined_text = f"{context_text}\n\n### FULL DIALOGUE\n\n{full_dialogue}"
    tokens = tokenizer.encode(combined_text)
    
    if len(tokens) <= max_tokens:
        return combined_text
    
    # Si no cabe todo, usar solo el contexto estructurado + parte del diálogo
    tokens_for_dialogue = max_tokens - len(tokenizer.encode(context_text + "\n\n### PARTIAL DIALOGUE\n\n"))
    
    # Seleccionar partes importantes del diálogo
    important_rows = []
    importance_score = {}
    
    # Asignar puntuación de importancia a cada línea
    for i, row in enumerate(transcript_rows):
        score = 0
        text = row['text'].lower()
        
        # Menciones de nombre, experiencia, educación o habilidades tienen mayor importancia
        if any(keyword in text for keyword in ["name", "call", "experience", "work", "job", "degree", "education", "skill", "strength"]):
            score += 3
        
        # Respuestas largas suelen tener más información
        if len(text) > 100:
            score += 2
        
        # Líneas al principio y al final son importantes
        if i < 5 or i > len(transcript_rows) - 5:
            score += 2
            
        importance_score[i] = score
    
    # Ordenar filas por importancia y seleccionar hasta el límite de tokens
    sorted_rows = sorted(range(len(transcript_rows)), key=lambda i: importance_score[i], reverse=True)
    
    dialogue_tokens = 0
    for i in sorted_rows:
        row_text = f"[{transcript_rows[i]['timestamp_s']}s] {transcript_rows[i]['text']}"
        row_tokens = len(tokenizer.encode(row_text))
        
        if dialogue_tokens + row_tokens <= tokens_for_dialogue:
            important_rows.append((int(float(transcript_rows[i]['timestamp_s'])), row_text))
            dialogue_tokens += row_tokens
        
        if dialogue_tokens >= tokens_for_dialogue:
            break
    
    # Ordenar las filas seleccionadas por timestamp para mantener el orden cronológico
    important_rows.sort(key=lambda x: x[0])
    partial_dialogue = "\n".join([row[1] for row in important_rows])
    
    return f"{context_text}\n\n### PARTIAL DIALOGUE\n\n{partial_dialogue}"

def format_time(seconds: float) -> str:
    """Formatea segundos en formato hora:minuto:segundo"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

# ------------------------------------------------------------------------- #
# Helper functions - Report Generation & Validation
# ------------------------------------------------------------------------- #
def build_report_prompt(transcript_context: str, candidate_info: Dict[str, Any], interview_duration: float) -> str:
    """Construye un prompt detallado para el modelo con la información extraída"""
    # Formatear la información extraída
    name = candidate_info["name"]
    professional_exp = ", ".join(candidate_info["professional_experience"][:3]) if candidate_info["professional_experience"] else "Unspecified"
    academic_bg = ", ".join(candidate_info["academic_background"][:2]) if candidate_info["academic_background"] else "Unspecified"
    strengths = ", ".join(candidate_info["strengths"][:5]) if candidate_info["strengths"] else "Unspecified"
    skills = ", ".join(candidate_info["skills"][:7]) if candidate_info["skills"] else "Unspecified"
    position = candidate_info["position_applied"]
    
    # Evidencia de trabajo en equipo
    teamwork_evidence = "Yes, has mentioned teamwork experience" if candidate_info["teamwork"] else "Not explicitly mentioned"
    
    # Tiempo de entrevista formateado
    time_str = format_time(interview_duration)
    
    prompt = f"""You are a senior HR analyst. Create a professional interview report in GitHub-flavoured Markdown based on the transcript provided. The report should be concise but comprehensive, highlighting the candidate's background, skills, and suitability for the position.

I have already extracted some key information from the transcript:

- Candidate Name: {name}
- Position Applied: {position}
- Professional Experience: {professional_exp}
- Academic Background: {academic_bg}
- Identified Strengths: {strengths}
- Identified Skills: {skills}
- Teamwork Experience: {teamwork_evidence}
- Interview Duration: {time_str}

INSTRUCTIONS:
1. Create a professional report following EXACTLY this structure:
2. Verify the extracted information against the transcript and correct any inaccuracies
3. Write a comprehensive SUMMARY section (150-200 words) that captures the essence of the interview
4. In the SKILLS section, categorize and elaborate on the candidate's technical and soft skills that are mentioned in the transcript
5. Create specific ACTION ITEMS formatted as bullet points (Who — What — When) for the next steps that must be followed with these candidate

REQUIRED OUTPUT FORMAT:

Name: [Candidate Name]  
Professional Experience: [Concise overview of work history]  
Academic Background: [Education details]  
Strengths: [Key strengths, comma-separated]  
Teamwork Skills: [Assessment of teamwork abilities]  
Time: {time_str}  

## SUMMARY  
[Comprehensive 200-250 word summary of the interview, highlighting candidate's background, qualifications, and suitability for the position]

## SKILLS  
[List and brief assessment of technical and soft skills demonstrated]

## ACTION ITEMS  
- [Who] — [What] — [When]  
- [At least 3 specific follow-up actions with clear ownership]

TRANSCRIPT:
{transcript_context}
"""
    return prompt

def validate_report(report: str, candidate_info: Dict[str, Any], interview_duration: float) -> str:
    """Valida y corrige el formato e información del informe generado"""
    # Verificar estructura básica
    required_sections = ["Name:", "Professional Experience:", "Academic Background:", "Strengths:", 
                         "Teamwork Skills:", "Time:", "## SUMMARY", "## SKILLS", "## ACTION ITEMS"]
    
    for section in required_sections:
        if section not in report:
            # Corregir sección faltante
            if section == "Name:":
                name = candidate_info["name"]
                report = f"Name: {name}\n{report}"
            elif section == "Professional Experience:":
                experience = ", ".join(candidate_info["professional_experience"][:3]) if candidate_info["professional_experience"] else "Unspecified"
                report = report.replace("Name:", f"Name: \nProfessional Experience: {experience}")
            elif section == "Academic Background:":
                academic = ", ".join(candidate_info["academic_background"][:2]) if candidate_info["academic_background"] else "Unspecified"
                if "Professional Experience:" in report:
                    report = report.replace("Professional Experience:", f"Professional Experience: \nAcademic Background: {academic}")
                else:
                    report = report.replace("Name:", f"Name: \nAcademic Background: {academic}")
            elif section == "Strengths:":
                strengths = ", ".join(candidate_info["strengths"][:5]) if candidate_info["strengths"] else "adaptability, communication, problem-solving"
                if "Academic Background:" in report:
                    report = report.replace("Academic Background:", f"Academic Background: \nStrengths: {strengths}")
                else:
                    report = report.replace("Name:", f"Name: \nStrengths: {strengths}")
            elif section == "Teamwork Skills:":
                teamwork = "Demonstrated ability to work effectively in team environments" if candidate_info["teamwork"] else "Not explicitly assessed"
                if "Strengths:" in report:
                    report = report.replace("Strengths:", f"Strengths: \nTeamwork Skills: {teamwork}")
                else:
                    report = report.replace("Name:", f"Name: \nTeamwork Skills: {teamwork}")
            elif section == "Time:":
                time_str = format_time(interview_duration)
                if "Teamwork Skills:" in report:
                    report = report.replace("Teamwork Skills:", f"Teamwork Skills: \nTime: {time_str}")
                else:
                    report = report.replace("Name:", f"Name: \nTime: {time_str}")
            elif section == "## SUMMARY":
                report += "\n\n## SUMMARY\nThe interview covered the candidate's professional background, skills, and suitability for the position. The candidate demonstrated relevant experience and qualifications."
            elif section == "## SKILLS":
                report += "\n\n## SKILLS\n" + ", ".join(candidate_info["skills"][:7]) if candidate_info["skills"] else "Technical skills and interpersonal abilities not fully assessed during interview."
            elif section == "## ACTION ITEMS":
                report += "\n\n## ACTION ITEMS\n- HR — Schedule follow-up interview — 1 week\n- Interviewer — Complete evaluation form — 2 days\n- Candidate — Submit additional portfolio materials — 1 week"
    
    # Asegurar que los elementos de acción están formateados correctamente
    if "## ACTION ITEMS" in report:
        action_section = report.split("## ACTION ITEMS")[1].strip()
        if not action_section.startswith("-"):
            corrected_actions = "- HR — Schedule follow-up interview — 1 week\n- Interviewer — Complete evaluation form — 2 days\n- Candidate — Submit additional portfolio materials — 1 week"
            report = report.replace(action_section, corrected_actions)
    
    return report

# ------------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------------- #
def main() -> None:
    start_time = time.time()
    warnings.filterwarnings("ignore")
    set_seed(SEED)
    
    print("[AGENT] Initializing...")
    
    # Configuración para optimizar la GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                # Cuantización de 4-bit
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,   # Doble cuantización para mayor eficiencia
        bnb_4bit_quant_type="nf4",        # Normal Float 4
    )
    
    # Carga del tokenizer y modelo con optimizaciones
    print("[AGENT] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Asegurar que el tokenizer esté configurado correctamente
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Cargar el modelo con configuraciones optimizadas
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Leer y procesar la transcripción
    print("[AGENT] Processing transcript...")
    transcript_rows, metadata = read_transcript(TRANSCRIPT_CSV)
    
    # Extraer información del candidato
    print("[AGENT] Extracting candidate information...")
    candidate_info = extract_candidate_info(transcript_rows)
    
    # Segmentar la transcripción
    print("[AGENT] Segmenting transcript...")
    segments = segment_transcript(transcript_rows)
    
    # Preparar contexto optimizado
    print("[AGENT] Preparing optimized context...")
    transcript_context = prepare_transcript_context(
        transcript_rows, 
        candidate_info, 
        segments,
        tokenizer, 
        CTX_LIMIT
    )
    
    # Construir prompt con información extraída
    print("[AGENT] Building enhanced prompt...")
    prompt = build_report_prompt(
        transcript_context, 
        candidate_info, 
        metadata["interview_duration"]
    )
    
    # Configurar el pipeline de generación
    print("[AGENT] Setting up generation pipeline...")
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.2,           # Menor temperatura para respuestas más precisas
        top_p=0.9,
        top_k=40,
        max_new_tokens=MAX_NEW_TOK,
        repetition_penalty=1.1,
        do_sample=True,
    )
    
    # Generar el informe
    print("[AGENT] Generating comprehensive report...")
    output = gen(prompt)[0]["generated_text"]
    
    # Extraer el informe (todo lo que sigue después del prompt)
    report = output[len(prompt):].strip()
    
    # Validar y corregir el informe
    print("[AGENT] Validating and enhancing report...")
    final_report = validate_report(
        report, 
        candidate_info, 
        metadata["interview_duration"]
    )
    
    # Guardar el informe
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(final_report, encoding="utf-8")
    
    # Informar del tiempo de ejecución
    elapsed_time = time.time() - start_time
    print(f"[AGENT] Report successfully generated and saved to {REPORT_PATH.resolve()}")
    print(f"[AGENT] Execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
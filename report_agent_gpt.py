# ------------------------------------------------------------------------- #
# GPU Utilities
# ------------------------------------------------------------------------- #
def check_gpu_availability() -> Tuple[bool, str]:
    """Verifica la disponibilidad de GPU y devuelve información sobre ella"""
    if not torch.cuda.is_available():
        return False, "GPU no disponible - se usará CPU (mucho más lento)"
    
    # Obtener información sobre la GPU
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    return True, f"GPU: {gpu_name} ({gpu_memory:.2f} GB)"

def get_gpu_memory_usage() -> Optional[float]:
    """Obtiene el uso actual de memoria en la GPU en GB"""
    if not torch.cuda.is_available():
        return None
    
    # Obtener memoria reservada por torch
    memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
    return memory_allocated

def optimize_for_ada_gpu() -> dict:
    """Configura parámetros optimizados para GPUs Ada (RTX 5000 series)"""
    # Configuración específica para Ada
    config = {
        "load_in_4bit": True,
        "compute_dtype": torch.float16,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16,
    }
    
    # Si hay poca memoria disponible, usar configuración más conservadora
    has_gpu, gpu_info = check_gpu_availability()
    if has_gpu and "RTX" in gpu_info and torch.cuda.get_device_properties(0).total_memory < 8 * (1024**3):
        # Si la GPU tiene menos de 8GB, usar una configuración más ligera
        config["load_in_8bit"] = True
        config["load_in_4bit"] = False
    
    return config

"""
report_agent.py — interview‑report generator with instruction-tuned model
=================================================================
• Lee data/transcripts.csv y crea un informe compacto en Markdown.
• Usa un modelo instruction-tuned para mejorar la calidad de respuesta.
• Optimizado para NVIDIA RTX 500 Ada Generation Laptop GPU

Requisitos (instalación única):
  pip install -U transformers accelerate sentencepiece optimum peft tqdm
  
Nota: La primera ejecución descargará el modelo (solo una vez).
      Las ejecuciones posteriores serán mucho más rápidas.
"""

from __future__ import annotations
import os, csv, re, warnings, time, sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from tqdm import tqdm
from huggingface_hub import HfApi
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    pipeline, set_seed, logging as tf_logging
)

# ------------------------------------------------------------------------- #
# Config
# ------------------------------------------------------------------------- #
TRANSCRIPT_CSV = Path("data/transcripts.csv")
REPORT_PATH    = Path("data/conversation_report.md")
CACHE_DIR      = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

# Modelo más pequeño pero instruction-tuned que funciona bien en tu GPU
MODEL_NAME     = "microsoft/Phi-3-mini-4k-instruct"  # Alternativa: "google/gemma-2b-it"
CTX_LIMIT      = 2500      # Ventana de contexto más grande
MAX_NEW_TOK    = 600       # Más tokens para respuestas completas
SEED           = 42        # Para reproducibilidad
FIRST_RUN_MSG  = True      # Mostrar mensaje de primera ejecución

# ------------------------------------------------------------------------- #
# Helper functions
# ------------------------------------------------------------------------- #
def read_transcript(p: Path) -> tuple[str, Dict[str, str]]:
    """Lee el CSV y extrae datos estructurados y el texto de la transcripción"""
    if not p.exists():
        raise FileNotFoundError("Run AudioTranscriber first → data/transcripts.csv")
    
    transcript_text = ""
    metadata = {
        "candidate_name": "Unspecified",
        "roles": set()
    }
    
    with p.open(encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
        
        # Extraer metadata
        for r in rows:
            text = r['text']
            # Detectar posibles nombres de candidatos
            if "name is" in text.lower() or "i am" in text.lower() or "i'm" in text.lower():
                # Extraer nombres propios
                name_match = re.search(r"(?:name is|I am|I'm) ([A-Z][a-z]+ [A-Z][a-z]+)", text)
                if name_match:
                    metadata["candidate_name"] = name_match.group(1)
            
            # Detectar roles en la conversación
            if ":" in text:
                role = text.split(":")[0].strip()
                metadata["roles"].add(role)
            
        # Crear texto de transcripción
        transcript_text = "\n".join(f"[{r['timestamp_s']}s] {r['text']}" for r in rows)
        
    return transcript_text, metadata

def clip_tokens(txt: str, tok, limit: int) -> str:
    """Recorta el texto si excede el límite de tokens, preservando contexto"""
    inputs = tok.encode(txt)
    if len(inputs) <= limit:
        return txt
    
    # Preservar el inicio y el final de la transcripción
    start_portion = int(limit * 0.3)
    end_portion = limit - start_portion
    preserved_tokens = inputs[:start_portion] + inputs[-end_portion:]
    
    return tok.decode(preserved_tokens, skip_special_tokens=True)

def extract_report_sections(text: str) -> Dict[str, str]:
    """Extrae las secciones del informe para validación"""
    sections = {}
    
    # Extraer los metadatos iniciales
    meta_pattern = r"Name: (.*?)\nExperience: (.*?)\nAcademic background: (.*?)\nStrengths: (.*?)\nOverall Impression: (.*?)\n"
    meta_match = re.search(meta_pattern, text, re.DOTALL)
    if meta_match:
        sections["metadata"] = {
            "name": meta_match.group(1).strip(),
            "experience": meta_match.group(2).strip(),
            "academic": meta_match.group(3).strip(),
            "strengths": meta_match.group(4).strip(),
            "impression": meta_match.group(5).strip()
        }
    
    # Extraer secciones principales
    main_sections = ["SUMMARY", "ANALYSIS", "ACTION ITEMS", "FIT & MOTIVATION", "EMOTIONAL TONE"]
    for i, section in enumerate(main_sections):
        pattern = rf"## {section}\s*(.*?)(?:(?=## )|$)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[section] = match.group(1).strip()
    
    return sections

def validate_and_fix_report(report: str) -> str:
    """Valida y corrige el formato del informe"""
    sections = extract_report_sections(report)
    
    # Plantilla de informe completo
    template = """Name: {name}  
Experience: {experience}  
Academic background: {academic}  
Strengths: {strengths}  
Overall Impression: {impression}

## SUMMARY  
{summary}

## ANALYSIS  
{analysis}

## ACTION ITEMS  
{action_items}

## FIT & MOTIVATION  
{fit_motivation}

## EMOTIONAL TONE  
{emotional_tone}
"""
    
    # Valores por defecto
    defaults = {
        "name": "Unspecified",
        "experience": "Unspecified",
        "academic": "Unspecified",
        "strengths": "adaptability, communication, problem-solving",
        "impression": "Requires further assessment.",
        "summary": "The interview covered the candidate's background and experience.",
        "analysis": "Insufficient information to make a complete assessment.",
        "action_items": "- HR — Follow up with candidate — 1 week\n- Interviewer — Complete evaluation form — 2 days\n- Candidate — Submit portfolio — 1 week",
        "fit_motivation": "Needs further exploration to determine cultural fit.",
        "emotional_tone": "Professional and composed."
    }
    
    # Fusionar datos extraídos con valores por defecto
    data = defaults.copy()
    
    if "metadata" in sections:
        for key, value in sections["metadata"].items():
            if value.strip():
                data[key] = value
    
    for section in ["SUMMARY", "ANALYSIS", "ACTION ITEMS", "FIT & MOTIVATION", "EMOTIONAL TONE"]:
        if section in sections and sections[section]:
            key = section.lower().replace(" & ", "_").replace(" ", "_")
            data[key] = sections[section]
    
    # Formatear los elementos de acción
    if not data["action_items"].startswith("-"):
        data["action_items"] = "- HR — Follow up — 1 week\n- Interviewer — Complete evaluation — 2 days\n- Candidate — Submit documents — 1 week"
    
    return template.format(**data)

# ------------------------------------------------------------------------- #
# Prompt avanzado con instrucciones claras
# ------------------------------------------------------------------------- #
PROMPT_TMPL = """You are a senior HR analyst. Produce a concise, professionally formatted interview report in GitHub‑flavoured Markdown based on the transcript provided.

STRICT rules:
1. DO NOT copy or quote the raw transcript text.
2. Fill every section; if unknown, write "Unspecified".
3. Keep the full report ≤ 450 words.
4. Use bullet lists only where indicated.
5. Extract specific information from the transcript: candidate name, experience, education, skills.
6. Format ACTION ITEMS as "Who — What — When" bullet points.

### OUTPUT FORMAT (exactly):

Name: <candidate name or "Unspecified">  
Experience: <concise work‑experience sentence>  
Academic background: <concise academic info>  
Strengths: <comma‑separated list of 3–5 strengths>  
Overall Impression: <one sentence>

## SUMMARY  
<≤180 words summary of the interview>

## ANALYSIS  
<brief critical analysis, ≤120 words>

## ACTION ITEMS  
- Who — What — When  
- <at least 3 bullet points>

## FIT & MOTIVATION  
<≤80 words on fit & motivation>

## EMOTIONAL TONE  
<one sentence, 2‑3 adjectives>

### TRANSCRIPT:
{transcript}
"""

# ------------------------------------------------------------------------- #
# MAIN
# ------------------------------------------------------------------------- #
def check_model_downloaded(model_name: str) -> bool:
    """Verifica si el modelo ya ha sido descargado previamente"""
    try:
        # Comprobar si existe la carpeta del modelo en la caché
        model_path = CACHE_DIR / "models--" + model_name.replace("/", "--")
        return model_path.exists() and any(model_path.glob("**/model*.safetensors"))
    except Exception:
        return False

def main() -> None:
    start_time = time.time()
    warnings.filterwarnings("ignore")
    # Reducir verbosidad de transformers
    tf_logging.set_verbosity_error()
    set_seed(SEED)
    
    # Verificar GPU
    has_gpu, gpu_info = check_gpu_availability()
    
    print("\n" + "="*80)
    if has_gpu:
        print(f" 🚀 {gpu_info}")
        print(" ✅ GPU detectada - procesamiento acelerado disponible")
    else:
        print(" ❌ GPU no detectada - usando CPU (procesamiento mucho más lento)")
        print(" 💡 Asegúrate de tener instalados los drivers NVIDIA y PyTorch con CUDA")
    print("="*80 + "\n")
    
    # Verificar si es la primera ejecución para este modelo
    is_first_run = not check_model_downloaded(MODEL_NAME)
    
    if is_first_run and FIRST_RUN_MSG:
        print("\n" + "="*80)
        print(f" PRIMERA EJECUCIÓN DETECTADA PARA {MODEL_NAME}")
        print(" Se descargarán archivos del modelo (varios GB) y se configurará el entorno.")
        print(" Esta descarga es ÚNICA - las futuras ejecuciones serán mucho más rápidas.")
        print(" Los archivos se guardarán en:", CACHE_DIR)
        print("="*80 + "\n")
    
    print("[AGENT] Initializing...")
    
    # Obtener configuración optimizada para tu GPU Ada
    gpu_config = optimize_for_ada_gpu()
    
    # Configuración para optimizar la GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=gpu_config.get("load_in_4bit", True),
        load_in_8bit=gpu_config.get("load_in_8bit", False),
        bnb_4bit_compute_dtype=gpu_config.get("bnb_4bit_compute_dtype", torch.float16),
        bnb_4bit_use_double_quant=gpu_config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=gpu_config.get("bnb_4bit_quant_type", "nf4"),
    )
    
    # Carga del tokenizer y modelo con optimizaciones
    print("[AGENT] Loading model and tokenizer...")
    
    # Mostrar barra de progreso para la primera descarga
    if is_first_run:
        with tqdm(total=100, desc="Downloading model files", unit="%") as pbar:
            def progress_callback(progress):
                pbar.update(int(progress * 100) - pbar.n)
            
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME, 
                cache_dir=CACHE_DIR
            )
            
            # Actualizar progreso
            pbar.update(30 - pbar.n)  # Tokenizer ~30%
            
            # Cargar el modelo con barra de progreso
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
                cache_dir=CACHE_DIR
            )
            pbar.update(100 - pbar.n)  # Completar barra
    else:
        # Carga normal sin barras de progreso para ejecuciones posteriores
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            cache_dir=CACHE_DIR
        )
        
        # Cargar el modelo con configuraciones optimizadas
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=CACHE_DIR
        )
    
    # Asegurar que el tokenizer esté configurado correctamente
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Leer y procesar la transcripción
    print("[AGENT] Processing transcript...")
    transcript_text, metadata = read_transcript(TRANSCRIPT_CSV)
    
    # Preparar el prompt con la transcripción recortada si es necesario
    transcript_clip = clip_tokens(transcript_text, tokenizer, CTX_LIMIT)
    prompt = PROMPT_TMPL.format(transcript=transcript_clip)
    
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
    print("[AGENT] Generating report...")
    output = gen(prompt)[0]["generated_text"]
    
    # Extraer el informe (todo lo que sigue después del prompt)
    report = output[len(prompt):].strip()
    
    # Si el modelo no generó un informe completo, verificar y corregir
    if "Name:" not in report:
        print("[AGENT] Warning: Model didn't generate proper report format. Extracting content...")
        # Intentar extraer contenido útil o usar una plantilla
        if len(report) > 50:  # Hay contenido sustancial
            report = f"Name: {metadata.get('candidate_name', 'Unspecified')}  \n" + report
        else:
            print("[AGENT] Generating minimal report based on transcript...")
            # Generar reporte mínimo basado en metadata extraída
            report = f"""Name: {metadata.get('candidate_name', 'Unspecified')}  
Experience: Unspecified  
Academic background: Unspecified  
Strengths: communication, adaptability, professionalism  
Overall Impression: Requires additional assessment.

## SUMMARY  
The interview covered the candidate's background and experience.

## ANALYSIS  
Insufficient information for complete assessment.

## ACTION ITEMS  
- HR — Schedule follow-up interview — 1 week
- Interviewer — Complete evaluation form — 2 days
- Candidate — Submit portfolio — 1 week

## FIT & MOTIVATION  
Requires further exploration to determine cultural fit.

## EMOTIONAL TONE  
Professional and composed.
"""
    
    # Validar y corregir el informe generado
    print("[AGENT] Validating and fixing report format...")
    final_report = validate_and_fix_report(report)
    
    # Guardar el informe
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(final_report, encoding="utf-8")
    
    # Informar del tiempo de ejecución
    elapsed_time = time.time() - start_time
    print(f"[AGENT] Report saved to {REPORT_PATH.resolve()}")
    print(f"[AGENT] Execution time: {elapsed_time:.2f} seconds")
    
    # Mostrar uso final de GPU
    if has_gpu:
        final_memory = get_gpu_memory_usage()
        print(f"[AGENT] Final GPU memory usage: {final_memory:.2f} GB")
        
        # Limpiar memoria GPU explícitamente
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[AGENT] GPU memory cache cleared")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[AGENT] Process interrupted by user")
        # Limpiar memoria GPU al salir
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")
        # En caso de error, intentar liberar recursos
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
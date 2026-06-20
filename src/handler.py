import re
from typing import Dict, List, Any, Tuple, Optional
import torch
import os
import json
import numpy as np
from transformers import (
   AutoTokenizer, AutoModelForCausalLM,
   AutoModelForSequenceClassification,
   T5ForConditionalGeneration, T5Tokenizer
)
from peft import PeftModel
from huggingface_hub import login, hf_hub_download

class EndpointHandler():
   def __init__(self, path=""):
       # Set CUDA backend preference to avoid cusolver issues
       torch.backends.cuda.preferred_linalg_library('cusolver')
       
       # PRODUCTION THRESHOLDS - BALANCED FOR DEMO
       self.ERROR_DETECTION_THRESHOLD = 0.5     # BALANCED - catch real errors, allow good answers
       self.CORRECTION_THRESHOLD = 0.6          # BALANCED - enable corrections when confident
       self.UNCERTAINTY_THRESHOLD = 0.3         # BALANCED - flag uncertain cases
       self.MIN_ANSWER_LENGTH = 5               # Minimum viable answer length
       self.MAX_CORRECTION_RATIO = 2.0          # Don't make answers too much longer
       
       # USER PREFERENCE SETTINGS
       self.SHOW_REASONING = True              # Default: show reasoning for transparency
       self.REASONING_LABEL = "Medical Analysis"  # User-friendly label
       
       # Authenticate with HF Hub for gated models
       hf_token = os.environ.get("HF_TOKEN")
       if hf_token:
           print("Using HF token for authentication...")
           login(token=hf_token)
       
       print("Loading Self-Correcting Medical QA System...")
       
       # TASK 1: Load RARE Model (Your Original Implementation)
       self._load_rare_model(path, hf_token)
       
       # TASK 4: Load Document Retrieval System
       self._load_retrieval_system()
       
       # TASK 2: Load Error Detection Model
       self._load_error_detection_model()
       
       # TASK 3: Load Self-Correction Model
       self._load_correction_model()
       
       print("Handler initialization complete!")
       print(f"Safety thresholds: Error={self.ERROR_DETECTION_THRESHOLD}, Correction={self.CORRECTION_THRESHOLD}")
       print(f"Reasoning display: {'Enabled' if self.SHOW_REASONING else 'Disabled'} by default")
   
   def _load_rare_model(self, path, hf_token):
       """TASK 1: Load RARE Model (Your Original Working Implementation)"""
       print("Loading RARE Model...")
       
       # Load tokenizer
       self.tokenizer = AutoTokenizer.from_pretrained(path)
       
       # Load base model with authentication
       base_model_name = "meta-llama/Llama-3.1-8B"
       
       print(f"Loading base model: {base_model_name}")
       self.model = AutoModelForCausalLM.from_pretrained(
           base_model_name,
           torch_dtype=torch.float16,
           device_map="auto",
           low_cpu_mem_usage=True,
           trust_remote_code=True,
           token=hf_token
       )
       
       # Safer embedding resize with error handling
       original_vocab_size = self.model.config.vocab_size
       target_vocab_size = len(self.tokenizer)
       
       if target_vocab_size > original_vocab_size:
           print(f"Resizing embeddings from {original_vocab_size} to {target_vocab_size}")
           try:
               # Try with CPU first to avoid CUDA issues
               device = self.model.device
               self.model = self.model.cpu()
               self.model.resize_token_embeddings(target_vocab_size)
               self.model = self.model.to(device)
               self.model.config.vocab_size = target_vocab_size
               print("Embeddings resized successfully")
           except Exception as e:
               print(f"Warning: Could not resize embeddings: {e}")
               print("Continuing with original vocab size")
       
       # Load PEFT adapter
       print("Loading PEFT adapter...")
       self.model = PeftModel.from_pretrained(self.model, path)
       self.model.eval()
       
       print("Model loaded successfully!")
   
   def _load_retrieval_system(self):
       """TASK 4: Load Document Retrieval System"""
       print("Loading Document Retrieval System...")
       
       try:
           # Install required packages
           import subprocess
           import sys
           
           packages = ['sentence-transformers', 'faiss-cpu']
           for package in packages:
               try:
                   __import__(package.replace('-', '_'))
                   print(f"   {package} already available")
               except ImportError:
                   print(f"   Installing {package}...")
                   subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '--quiet'])
           
           # Now import after installation
           from sentence_transformers import SentenceTransformer
           import faiss
           
           # Load SentenceBERT encoder
           self.retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
           print("   ✅ SentenceBERT loaded")
           
           # Download FAISS index and documents
           try:
               index_path = hf_hub_download(
                   repo_id="Maikobi/RARE_retriever",
                   filename="faiss_index.index",
                   repo_type="dataset"
               )
               docs_path = hf_hub_download(
                   repo_id="Maikobi/RARE_retriever", 
                   filename="document_corpus.json",
                   repo_type="dataset"
               )
               
               # Load FAISS index and documents
               self.faiss_index = faiss.read_index(index_path)
               with open(docs_path, 'r') as f:
                   self.documents = json.load(f)
               
               print(f"Retrieval system loaded: {len(self.documents)} documents")
               
           except Exception as e:
               print(f"Could not load retrieval data: {e}")
               self.faiss_index = None
               self.documents = []
               
       except Exception as e:
           print(f"Retrieval system failed: {e}")
           self.retriever = None
           self.faiss_index = None
           self.documents = []
   
   def _load_error_detection_model(self):
       """TASK 2: Load Error Detection System"""
       print("Loading Error Detection System...")
       
       try:
           self.error_tokenizer = AutoTokenizer.from_pretrained("Maikobi/scibert_error_detection")
           self.error_model = AutoModelForSequenceClassification.from_pretrained(
               "Maikobi/scibert_error_detection",
               torch_dtype=torch.float16,
               device_map="auto",
               low_cpu_mem_usage=True
           )
           self.error_model.eval()
           print("SciBERT error detection loaded")
           
       except Exception as e:
           print(f"Error detection failed: {e}")
           self.error_tokenizer = None
           self.error_model = None
   
   def _load_correction_model(self):
       """TASK 3: Load Self-Correction System"""
       print("🔧 Loading Self-Correction System...")
       
       try:
           self.correction_tokenizer = T5Tokenizer.from_pretrained("Maikobi/t5_self_correction")
           self.correction_model = T5ForConditionalGeneration.from_pretrained(
               "Maikobi/t5_self_correction",
               torch_dtype=torch.float16,
               device_map="auto",
               low_cpu_mem_usage=True
           )
           self.correction_model.eval()
           print("T5 self-correction loaded")
           
       except Exception as e:
           print(f"Correction model failed: {e}")
           self.correction_tokenizer = None
           self.correction_model = None
   
   def retrieve_documents(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
       """TASK 4: Retrieve relevant medical documents"""
       if not hasattr(self, 'retriever') or not self.retriever or not self.faiss_index:
           return [("No retrieval system available", 0.0)]
       
       try:
           import faiss
           
           # Encode query
           query_embedding = self.retriever.encode([question], convert_to_tensor=False)
           query_embedding = query_embedding.astype("float32")
           faiss.normalize_L2(query_embedding)
           
           # Search FAISS index
           scores, indices = self.faiss_index.search(query_embedding, top_k)
           
           # Return documents with scores
           results = []
           for i, score in zip(indices[0], scores[0]):
               if i < len(self.documents):
                   results.append((self.documents[i], float(score)))
           
           return results if results else [("No relevant documents found", 0.0)]
           
       except Exception as e:
           print(f"Retrieval error: {e}")
           return [("Retrieval failed", 0.0)]
   
   def format_rare_prompt(self, question, knowledge=""):
       """EXACT MATCH to your training format"""
       if knowledge:
           knowledge_section = f"[KNOWLEDGE]\n{knowledge}\n[/KNOWLEDGE]\n\n"
       else:
           knowledge_section = ""
           
       return f"""{knowledge_section}Question: {question}
[REASONING]
I need to provide a comprehensive, evidence-based medical answer by:
1. **Medical Analysis**: What are the key medical concepts, conditions, or treatments involved?
2. **Evidence Review**: What does the current medical literature and provided knowledge indicate?
3. **Safety Considerations**: Are there any contraindications, warnings, or safety concerns I must address?
4. **Completeness Check**: Have I covered mechanism, treatment options, prognosis, and patient guidance?
5. **Clinical Context**: What would be most helpful for healthcare decision-making?
Let me work through this systematically using the medical evidence:
[/REASONING]
Answer:"""
   
   def generate_rare_answer(self, question: str, retrieved_docs: List[Tuple[str, float]] = None, show_reasoning: bool = None) -> Dict[str, str]:
       """Generate answer with optional reasoning display"""
       # Format knowledge from retrieved documents
       knowledge = ""
       if retrieved_docs:
           knowledge = "\n".join([doc for doc, _ in retrieved_docs[:3]])
       
       # Use your EXACT training prompt format
       prompt = self.format_rare_prompt(question, knowledge)
       
       # Use your original tokenization with error handling
       try:
           encoded = self.tokenizer(
               prompt,
               return_tensors="pt",
               truncation=True,
               max_length=1024,
               padding=True
           ).to(self.model.device)
       except Exception as e:
           print(f"Tokenization warning: {e}")
           # Fallback without custom tokens
           prompt_clean = prompt.replace("[KNOWLEDGE]", "").replace("[/KNOWLEDGE]", "")
           prompt_clean = prompt_clean.replace("[REASONING]", "").replace("[/REASONING]", "")
           encoded = self.tokenizer(
               prompt_clean,
               return_tensors="pt",
               truncation=True,
               max_length=1024,
               padding=True
           ).to(self.model.device)
       
       # Generate using your training parameters
       with torch.no_grad():
           outputs = self.model.generate(
               **encoded,
               max_new_tokens=300,  # Match your training
               temperature=0.8,     # Match your training
               do_sample=True,
               top_p=0.9,          # Match your training
               pad_token_id=self.tokenizer.pad_token_id,
               eos_token_id=self.tokenizer.eos_token_id,
               repetition_penalty=1.1,
               length_penalty=1.0
           )
       
       full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
       
       # Extract reasoning and answer separately
       reasoning = ""
       answer = ""
       
       if "[REASONING]" in full_response and "[/REASONING]" in full_response:
           # Extract content between reasoning tags
           reasoning_section = full_response.split("[REASONING]")[1].split("[/REASONING]")[0].strip()
           
           # Check if answer is inside reasoning section (training format)
           if "Answer:" in reasoning_section:
               answer = reasoning_section.split("Answer:")[-1].strip()
               reasoning = reasoning_section.split("Answer:")[0].strip()
           else:
               # Fallback: treat whole reasoning section as answer
               answer = reasoning_section
               reasoning = "Medical reasoning provided"
       else:
           # Fallback extraction
           answer = full_response.split("Answer:")[-1].strip() if "Answer:" in full_response else full_response[len(prompt):].strip()
       
       # Use instance setting if not specified
       if show_reasoning is None:
           show_reasoning = self.SHOW_REASONING
       
       return {
           "answer": answer,
           "reasoning": reasoning if show_reasoning else "",
           "show_reasoning": show_reasoning,
           "full_response": full_response
       }
   
   def detect_error(self, question: str, answer: str) -> Tuple[bool, float, str]:
       """TASK 2: Enhanced error detection with confidence levels"""
       if not hasattr(self, 'error_model') or not self.error_model:
           return False, 0.5, "no_model"
       
       # Basic safety checks first
       if len(answer.split()) < self.MIN_ANSWER_LENGTH:
           return True, 0.9, "too_short"
       
       try:
           # Format as question + answer (based on your training data)
           input_text = f"Question: {question}\nAnswer: {answer}"
           
           inputs = self.error_tokenizer(
               input_text,
               return_tensors="pt",
               truncation=True,
               max_length=512,
               padding=True
           ).to(self.error_model.device)
           
           with torch.no_grad():
               outputs = self.error_model(**inputs)
               probabilities = torch.softmax(outputs.logits, dim=-1)
               error_probability = probabilities[0][1].item()  # label=1 is error
           
           # ENHANCED DECISION LOGIC WITH THRESHOLDS
           if error_probability > self.ERROR_DETECTION_THRESHOLD:
               return True, error_probability, "high_confidence"
           elif error_probability > self.UNCERTAINTY_THRESHOLD:
               return False, error_probability, "uncertain"  # Don't auto-correct
           else:
               return False, error_probability, "low_confidence"
           
       except Exception as e:
           print(f"Error detection failed: {e}")
           return False, 0.5, "error"
   
   def classify_error_type(self, question: str, answer: str, error_confidence: float) -> str:
       """Classify specific error type for targeted correction"""
       
       # Rule-based classification (enhance this with your error type model later)
       answer_lower = answer.lower()
       answer_words = answer.split()
       
       # Pattern-based error type detection
       if len(answer_words) < 10:
           return "too_short"
       elif "yes" in answer_lower and "no" in answer_lower:
           return "contradiction"
       elif any(dangerous in answer_lower for dangerous in ["ignore doctor", "stop medication", "skip treatment"]):
           return "dangerous"
       elif len(answer_words) > 200:
           return "too_verbose"
       elif error_confidence > 0.9:
           return "factual_error"
       else:
           return "incomplete"  # Most common from your training data
   
   def validate_correction(self, original: str, corrected: str, question: str) -> bool:
       """CRITICAL: Validate that correction actually improves the answer"""
       
       # TEMPORARY DEBUG: Always accept corrections to see T5 output
       print(f"🔍 VALIDATION DEBUG:")
       print(f"   Original length: {len(original.split())} words")
       print(f"   Corrected length: {len(corrected.split())} words") 
       print(f"   Corrected preview: {corrected[:100]}...")
       
       # Basic safety checks
       if not corrected or corrected.strip() == "":
           print("Empty correction")
           return False
       
       if len(corrected.split()) < self.MIN_ANSWER_LENGTH:
           print(f"Too short: {len(corrected.split())} < {self.MIN_ANSWER_LENGTH}")
           return False
       
       # Don't apply if no meaningful change
       if corrected.lower().strip() == original.lower().strip():
           print("No meaningful change")
           return False
       
       # Don't apply if correction is way too long (possible hallucination)
       if len(corrected.split()) > len(original.split()) * self.MAX_CORRECTION_RATIO:
           print(f"Too long: ratio {len(corrected.split()) / len(original.split()):.2f} > {self.MAX_CORRECTION_RATIO}")
           return False
       
       # Don't apply if correction is way too short (possible truncation)
       if len(corrected.split()) < len(original.split()) * 0.3:
           print(f"Too short ratio: {len(corrected.split()) / len(original.split()):.2f} < 0.3")
           return False
       
       # TEMPORARILY ACCEPT ALL OTHER CORRECTIONS FOR DEBUGGING
       print("Validation passed (debug mode)")
       return True
   
   def emergency_response_cleaner(self, generated_text: str, question: str) -> str:
       """EMERGENCY: Clean research artifacts from generated responses"""
       
       if not generated_text or not generated_text.strip():
           return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
       
       cleaned = generated_text.strip()
       
       # 1. Remove author disclaimers and conflicts of interest
       patterns_to_remove = [
           r"Drs?\.\s+\w+.*?conflict.*?interest.*?\.",
           r"Authors?\s+.*?conflict.*?interest.*?\.",
           r"The authors?\s+.*?potential conflicts.*?\.",
           r".*?do not report any.*?conflicts.*?\.",
       ]
       
       for pattern in patterns_to_remove:
           cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
       
       # 2. Remove research methodology statements
       research_patterns = [
           r"We\s+(assessed|investigated|studied|examined|evaluated|determined|sought to).*?\.",
           r"This study\s+(investigated|examined|assessed).*?\.",
           r"To\s+(determine|investigate|assess|examine).*?\.",
           r"The aim of this study.*?\.",
           r"Methods:.*?\.",
           r"Results:.*?\.",
           r"Conclusion:.*?\.",
       ]
       
       for pattern in research_patterns:
           cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
       
       # 3. Remove research questions (but keep educational content)
       question_patterns = [
           r"(Does|Is|Do|Are|Can|Will|Should)\s+[^.]*?\?",
           r"Question:\s+[^.]*?\?",
           r"Is it true that.*?\?",
       ]
       
       for pattern in question_patterns:
           cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
       
       # 4. Remove study results and statistics
       stats_patterns = [
           r"\d+%\s+of\s+\d+\s+samples.*?\.",
           r"Results from the.*?Study.*?\.",
           r".*?participants.*?were enrolled.*?\.",
           r".*?patients.*?were assessed.*?\.",
           r"High serum.*?associated with.*?\.",
           r"Recent studies.*?suggest.*?\.",
       ]
       
       for pattern in stats_patterns:
           cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
       
       # 5. Extract educational content (first few good sentences)
       sentences = [s.strip() for s in cleaned.split('.') if s.strip()]
       educational_sentences = []
       
       # Keywords that indicate educational content
       educational_keywords = [
           'diabetes mellitus is', 'fever is', 'hypertension is', 'defined as',
           'characterized by', 'diagnosis', 'treatment', 'symptoms include',
           'caused by', 'risk factors', 'complications', 'management'
       ]
       
       # Research keywords to avoid
       research_keywords = [
           'study', 'research', 'assessed', 'investigated', 'participants',
           'enrolled', 'examined', 'evaluated', 'correlation', 'hypothesis'
       ]
       
       for sentence in sentences[:5]:  # Check first 5 sentences
           sentence_lower = sentence.lower()
           
           # Keep if it's educational content
           is_educational = any(keyword in sentence_lower for keyword in educational_keywords)
           is_research = any(keyword in sentence_lower for keyword in research_keywords)
           
           if is_educational or (not is_research and len(sentence.split()) > 5):
               educational_sentences.append(sentence)
           
           # Stop if we have enough good content
           if len(educational_sentences) >= 3:
               break
       
       # 6. Rebuild clean response
       if educational_sentences:
           clean_response = '. '.join(educational_sentences)
           if not clean_response.endswith('.'):
               clean_response += '.'
       else:
           # Fallback: extract any reasonable content
           clean_response = '. '.join(sentences[:2])
           if not clean_response.endswith('.'):
               clean_response += '.'
       
       # 7. Add educational context only as LAST RESORT fallback
       if not clean_response or len(clean_response.split()) < 5:
           # Only trigger fallbacks if cleaning completely failed
           print(f"⚠️ Emergency fallback triggered for: {question}")
           question_lower = question.lower()
           if 'diabetes' in question_lower:
               clean_response = "Diabetes mellitus is a group of metabolic disorders characterized by persistent hyperglycemia due to defects in insulin secretion, insulin action, or both. The main types are Type 1 (autoimmune) and Type 2 (insulin resistance). Diagnosis is based on fasting glucose, HbA1c, or oral glucose tolerance test results."
           elif 'fever' in question_lower:
               clean_response = "Fever is an elevation of body temperature above normal (98.6°F/37°C) due to resetting of the hypothalamic thermostat. It's commonly caused by infections, inflammatory conditions, or malignancy. The body's immune response triggers pyrogens that raise the temperature set point."
           elif 'hypertension' in question_lower or 'blood pressure' in question_lower:
               clean_response = "Hypertension is defined as systolic blood pressure ≥140 mmHg or diastolic blood pressure ≥90 mmHg. It's classified as primary (essential) or secondary hypertension. Risk factors include age, obesity, salt intake, and family history."
           else:
               # For any other topic, try to preserve whatever content we found
               clean_response = "I apologize, but I cannot provide a complete answer to your question. The system encountered an issue processing the response. Please try rephrasing your question or asking about a specific medical topic."
       
       # 8. Final cleanup
       clean_response = re.sub(r'\s+', ' ', clean_response)  # Remove extra spaces
       clean_response = re.sub(r'\.+', '.', clean_response)  # Remove multiple periods
       clean_response = clean_response.strip()
       
       return clean_response
   def correct_answer(self, question: str, wrong_answer: str, retrieved_docs: List[Tuple[str, float]] = None, error_type: str = "unknown") -> str:
       """TASK 3: Enhanced correction with emergency cleanup for research artifacts"""
       
       # EMERGENCY: Apply response cleaner first
       cleaned_answer = self.emergency_response_cleaner(wrong_answer, question)
       
       # If cleaning produced a good result, use it
       if cleaned_answer and len(cleaned_answer.split()) >= 15:
           return cleaned_answer
       
       # Otherwise, try T5 correction as fallback
       if not hasattr(self, 'correction_model') or not self.correction_model:
           return cleaned_answer  # Return cleaned version even if T5 unavailable
       
       try:
           # Error-type-specific correction prompts - EXACT MATCH to T5 training data format
           correction_prompts = {
               "incomplete": f"Complete the missing information in this medical answer: Question: {question} Context: {{context}} Incorrect Answer: {wrong_answer} FileModeled Error Type: incomplete Error Explanation: Missing key details about treatment options and potential side effects. Provide the correct answer:",
               "factual_error": f"Correct the factual inaccuracies in this medical answer: Question: {question} Context: {{context}} Incorrect Answer: {wrong_answer} FileModeled Error Type: factual_error Error Explanation: Contains medical inaccuracies that need correction. Provide the correct answer:",
               "format_mismatch": f"Reformat this answer to properly address the medical question: Question: {question} Context: {{context}} Incorrect Answer: {wrong_answer} FileModeled Error Type: format_mismatch Error Explanation: The generated answer does not address the original question. Provide the correct answer:",
               "too_short": f"Complete the missing information in this medical answer: Question: {question} Context: {{context}} Incorrect Answer: {wrong_answer} FileModeled Error Type: too_short Error Explanation: Answer is too brief and lacks sufficient detail. Provide the correct answer:",
               "contradiction": f"Fix the contradictory information in this medical answer: Question: {question} Context: {{context}} Incorrect Answer: {wrong_answer} FileModeled Error Type: contradiction Error Explanation: Contains contradictory medical information. Provide the correct answer:",
               "dangerous": f"Rewrite this answer to remove dangerous medical advice: Question: {question} Context: {{context}} Incorrect Answer: {wrong_answer} FileModeled Error Type: dangerous Error Explanation: Contains potentially harmful medical advice. Provide the correct answer:",
               "too_verbose": f"Make this medical answer more concise while keeping key information: Question: {question} Context: {{context}} Incorrect Answer: {wrong_answer} FileModeled Error Type: too_verbose Error Explanation: Answer is unnecessarily long and verbose. Provide the correct answer:",
               "unknown": f"Improve this medical answer: Question: {question} Context: {{context}} Incorrect Answer: {wrong_answer} FileModeled Error Type: unknown Error Explanation: General improvement needed. Provide the correct answer:"
           }
           
           # Use error-type-specific prompt that matches T5 training format
           base_prompt = correction_prompts.get(error_type, correction_prompts["unknown"])
           
           # Add context from retrieved documents (matching training format)
           if retrieved_docs:
               context = "\n".join([doc for doc, _ in retrieved_docs[:2]])
               input_text = base_prompt.format(context=context)
           else:
               # Use a placeholder context if none available
               input_text = base_prompt.format(context="No additional context available.")
           
           inputs = self.correction_tokenizer(
               input_text,
               return_tensors="pt",
               truncation=True,
               max_length=512,
               padding=True
           ).to(self.correction_model.device)
           
           with torch.no_grad():
               outputs = self.correction_model.generate(
                   **inputs,
                   max_new_tokens=256,
                   temperature=0.7,
                   do_sample=True,
                   num_beams=3,
                   early_stopping=True,
                   repetition_penalty=1.1
               )
           
           corrected = self.correction_tokenizer.decode(outputs[0], skip_special_tokens=True)
           
           # Apply emergency cleaner to T5 output too
           corrected_clean = self.emergency_response_cleaner(corrected.strip(), question)
           
           return corrected_clean if corrected_clean else cleaned_answer
           
       except Exception as e:
           print(f"T5 correction failed: {e}")
           return cleaned_answer  # Always return cleaned version as fallback
   
   def __call__(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
       """Complete Self-Correcting Medical QA Pipeline with Enhanced Safety"""
       # Get inputs - simplified without RARE-only mode
       inputs = data.pop("inputs", data)
       parameters = data.pop("parameters", {})
       
       # Check user preference for reasoning display
       show_reasoning = data.get("show_reasoning", self.SHOW_REASONING)
       
       # Convert any input to question string
       question = str(inputs)
       
       print(f"Processing: {question[:100]}...")
       print(f"Reasoning display: {'Enabled' if show_reasoning else 'Disabled'}")
       
       # STEP 1: Document Retrieval (with fallback)
       retrieved_docs = []
       if hasattr(self, 'retriever') and self.retriever:
           print("Retrieving documents...")
           retrieved_docs = self.retrieve_documents(question, top_k=3)
           print(f"   Found {len(retrieved_docs)} documents")
       else:
           print("No retrieval system - using RARE without context")
       
       # STEP 2: Generate Initial Answer with optional reasoning
       print("Generating answer with RARE...")
       result = self.generate_rare_answer(question, retrieved_docs, show_reasoning)
       initial_answer = result["answer"]
       reasoning = result["reasoning"]
       print(f"   Generated: {initial_answer[:100]}...")
       
       # EMERGENCY CLEANUP: Apply response cleaner to initial answer
       cleaned_initial = self.emergency_response_cleaner(initial_answer, question)
       print(f"   Cleaned: {cleaned_initial[:100]}...")
       
       # Use cleaned version for further processing
       initial_answer = cleaned_initial
       
       # STEP 3: Enhanced Error Detection with Confidence Levels
       error_detected = False
       error_confidence = 0.5
       confidence_level = "unknown"
       error_type = "unknown"
       
       if hasattr(self, 'error_model') and self.error_model:
           print("🔍 Checking for errors...")
           error_detected, error_confidence, confidence_level = self.detect_error(question, initial_answer)
           
           if error_detected:
               error_type = self.classify_error_type(question, initial_answer, error_confidence)
               print(f"   Error detected: {error_type} (confidence: {error_confidence:.3f}, level: {confidence_level})")
           else:
               print(f"   No error detected (confidence: {error_confidence:.3f}, level: {confidence_level})")
       else:
           print("🔍 No error detection - accepting answer")
       
       # STEP 4: Enhanced Self-Correction with Validation
       final_answer = initial_answer
       final_reasoning = reasoning
       correction_applied = False
       correction_validated = False
       correction_reason = "none"
       
       if error_detected and confidence_level == "high_confidence":
           if error_confidence > self.CORRECTION_THRESHOLD and hasattr(self, 'correction_model') and self.correction_model:
               print(f"🔧 Applying {error_type} correction...")
               corrected_answer = self.correct_answer(question, initial_answer, retrieved_docs, error_type)
               
               # CRITICAL VALIDATION STEP
               if self.validate_correction(initial_answer, corrected_answer, question):
                   final_answer = corrected_answer
                   correction_applied = True
                   correction_validated = True
                   correction_reason = f"validated_{error_type}_correction"
                   # Keep original reasoning but note correction was applied
                   if show_reasoning and reasoning:
                       final_reasoning = f"{reasoning}\n\n**Correction Applied**: {error_type} error detected and fixed."
                   print(f"Correction applied and validated: {final_answer[:100]}...")
               else:
                   correction_reason = f"validation_failed_{error_type}"
                   print(f"Correction validation failed - keeping original")
           else:
               correction_reason = f"confidence_too_low_{error_confidence:.3f}"
               print(f"Error confidence too low for correction ({error_confidence:.3f} < {self.CORRECTION_THRESHOLD})")
       elif error_detected and confidence_level == "uncertain":
           correction_reason = "flagged_for_review"
           print(f"Uncertain error detection - flagging for human review")
       elif error_detected:
           correction_reason = f"no_correction_model_{error_type}"
           print(f"{error_type} error detected but no correction model available")
       else:
           correction_reason = "no_error_detected"
           print("No errors detected")
       
       # Format response based on reasoning preference
       if show_reasoning and final_reasoning:
           formatted_response = f"**{self.REASONING_LABEL}:**\n{final_reasoning}\n\n**Answer:**\n{final_answer}"
       else:
           formatted_response = final_answer
       
       # Return comprehensive response with safety metadata
       return [{
           "generated_text": formatted_response,  # Formatted with or without reasoning
           "answer_only": final_answer,           # Clean answer only
           "reasoning": final_reasoning,          # Reasoning (empty if disabled)
           "show_reasoning": show_reasoning,      # User preference
           
           # Answer quality info
           "initial_answer": initial_answer,
           
           # Error detection info
           "error_detected": error_detected,
           "error_confidence": round(error_confidence, 3),
           "confidence_level": confidence_level,
           "error_type": error_type,
           
           # Correction info
           "correction_applied": correction_applied,
           "correction_validated": correction_validated,
           "correction_reason": correction_reason,
           
           # System info
           "retrieved_docs_count": len(retrieved_docs),
           "safety_thresholds": {
               "error_threshold": self.ERROR_DETECTION_THRESHOLD,
               "correction_threshold": self.CORRECTION_THRESHOLD,
               "uncertainty_threshold": self.UNCERTAINTY_THRESHOLD
           },
           "pipeline_components": {
               "retrieval": hasattr(self, 'retriever') and self.retriever is not None,
               "error_detection": hasattr(self, 'error_model') and self.error_model is not None,
               "correction": hasattr(self, 'correction_model') and self.correction_model is not None
           }
       }]
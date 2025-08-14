from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import os
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
#from langchain.vectorstores import FAISS

# Add imports for HuggingFace models
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import json
import re
from typing import Optional
from typing import List, Optional, Literal, Dict, Any

from langchain.prompts import ChatPromptTemplate
import openai
from dotenv import load_dotenv
load_dotenv('src/.env')
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ['GROQ_API_KEY'] = GROQ_API_KEY






# Model Selector class with class-level flag for model selection
class ModelSelector:
    # Class-level flag to select the model
    # Options: "openai", "deepseek-v3-8b", "deepseek-v3-13b"
    MODEL_TYPE = "openai"
    
    @classmethod
    def get_llm(cls, model_name="gpt-4o-mini", temperature=0):
        """
        Get a language model based on the class-level MODEL_TYPE flag.
        
        Args:
            model_name: Specific model name (used for OpenAI models)
            temperature: Temperature setting for the model
            
        Returns:
            A language model instance
        """
        if cls.MODEL_TYPE == "openai":
            return ChatOpenAI(model=model_name, temperature=temperature)
        
        elif cls.MODEL_TYPE == "deepseek-8b":
            # Login to Hugging Face
            login(token="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") #add hftoken here

            # Check if GPU is available 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            # Define model name
            #model_name ="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

            # Load tokenizer
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # DeepSeek-V3 8B quantized model
            model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
                token="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",  #add hftoken here 
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        elif cls.MODEL_TYPE == "deepseek-qwen-32b":
            # Login to Hugging Face
            login(token="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")

            # Check if GPU is available 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            # Define model name (quantized version)
            model_name = "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16"

            # Load tokenizer
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load the model with appropriate settings for the quantized version
            print("Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True,
                token="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",
            )
            
            # Create the pipeline for text generation
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        elif cls.MODEL_TYPE == "deepseek-v3-13b":
            # DeepSeek-V3 13B quantized model
            model_id = "deepseek-ai/deepseek-coder-v3-instruct-13b-q4"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype="auto"
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        
        else:
            # Default to OpenAI if model type is not recognized
            print(f"Warning: Unrecognized model type '{cls.MODEL_TYPE}'. Defaulting to OpenAI.")
            return ChatOpenAI(model=model_name, temperature=temperature)

class Interpretation(BaseModel):
    para_id : str = Field(description="Id of the para")
    if_interpretation: bool = Field(description="If the text contains a legal interpretation")
    interpretation_phrases: list = Field(description="List of interpretation phrases")
    reason: str = Field(description="Brief explanation of why or why not given paragraph contains any legal interpretation")
# Define Pydantic models for our response structure
class LegislationMatch(BaseModel):
    case_law_term: str
    section_id: str
    legislation_term: str
    key_concept: str
    confidence: Literal["High", "Medium", "Low"]

class LegislationMatchInfo(BaseModel):
    is_match: bool
    match_reasoning: str
    matches: List[LegislationMatch]
class Application(BaseModel):
    """
    Represents the classification outcome for a given paragraph in a case law text,
    indicating whether and how the law is applied to the specific facts.
    """
    
    para_id: str = Field(
        description="Unique identifier for the paragraph in the case law text."
    )
    if_law_applied: bool = Field(
        description="Indicates whether this paragraph demonstrates an application of law to the facts."
    )
    application_of_law_phrases: list[str] = Field(
        description="Collection of phrases or sentences showing how legal principles are applied in this paragraph."
    )
    reason: str = Field(
        description="Short explanation outlining why this paragraph does or does not demonstrate an application of law."
    )
def get_batch_job(input_file):
    client = openai.OpenAI()
    batch_input_file = client.files.create(file=open(input_file, "rb"), purpose="batch")


    batch_input_file_id = batch_input_file.id
    batch_job = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    )
    return batch_job
def getLegalClassifierUsingJson(case_law_json_list, examples, temperature=0):
    system_prompt2 = """APPLICATION OF LAW DEFINITION:
    An application of law is where statutory legal provisions are applied to the specific facts of the case at hand. This goes beyond merely citing or discussing law in the abstract and without specific reference to the facts of the case at hand.

INDICATORS OF APPLICATION OF LAW:
    1. The judge **connects specific statutory legal provisions** to the specific factual circumstances.
    2. The text shows reasoning that explains **how the law resolves or addresses the unique facts**.
    3. The paragraph contains the judge's analysis leading to a conclusion **based on legal principles**.
    4. **Legal tests or criteria** are being applied to the case facts.

NOT APPLICATIONS OF LAW:
    1. Mere citations of statutes, cases, or legal principles without **application to facts**.
    2. Background procedural information or **case history**.
    3. Statements about jurisdiction or **general legal explanations**.
    4. Summaries of arguments made by parties without **judicial analysis**.
    5. Restatements of previous cases without **connecting them to current facts**.

APPROACH TO ANALYSIS:
    1. Read the paragraph carefully to understand the context.
    2. Divide the case law into sections: **Facts/Background/Introduction**, **Issues/Arguments**, and **Decision**. Focus on the **Decision** part.
    3. **Focus on paragraphs in the Decision section**, where the judge connects the facts to the application of law. Even if specific statutes are not explicitly cited, consider whether **legal principles** or **standards** are applied to the facts.
    4. Identify any **indirect references** to law, such as when the judge applies legal reasoning, **procedural standards**, or **discretionary legal tests**, even if a specific statute is not directly mentioned.
    5. Mark paragraphs as **if_law_applied=True** if they demonstrate an application of law as described above. Mark paragraphs as **False** if they do not demonstrate a clear application of law.

**Additional Considerations**:
- **Implied Application of Law**: Even when no specific statute is cited, legal reasoning such as **'in accordance with,' 'following the principles of,' or 'based on the legal test'** can indicate the application of law to the facts.
- **Decisions about Penalties, Sanctions, or Procedures**: Decisions about the imposition of penalties or legal outcomes based on procedural adherence may also imply the application of law even if the statute isn’t named.
- **Reasoning Process**: If the judge is **weighing the facts and applying legal reasoning** to determine the outcome (e.g., whether something was appropriate or justified within legal standards), that should be flagged as an application of law.

Example:
- In paragraphs where the decision involves penalty points, disqualification, or procedural decisions, even without the exact reference to a statute, if the reasoning is based on a **legal framework** or **standards**, it should be flagged as an application of law.
"""
    system_prompt = """
    You are analyzing paragraphs from a single UK case law to determine if they contain an application of law to specific facts.

    APPLICATION OF LAW DEFINITION:
    An application of law is where statutory legal provisions are applied to the specific facts of the case at hand. This goes beyond merely citing or discussing law in the abstract and without specific reference to the facts of the case at hand.


    INDICATORS OF APPLICATION OF LAW:
    1. The judge connects specific statutory legal provisions to the specific factual circumstances.  
    2. The text shows reasoning that explains how the law resolves or addresses the unique facts
    3. The paragraph contains the judge's analysis leading to a conclusion based on legal principles
    4. Legal tests or criteria are being applied to the case facts

    NOT APPLICATIONS OF LAW:
    1. Mere citations of statutes, cases, or legal principles without application to facts
    2. Background procedural information or case history
    3. Statements about jurisdiction or general legal explanations
    4. Summaries of arguments made by parties without judicial analysis
    5. Restatements of previous cases without connecting them to current facts

    For each paragraph, determine if it contains an application of law, identify the specific phrases showing application. Use the other paragraphs as a context for your decision. If the whole case law is given it mostly have three sections: Facts/Background/introduction, Issues/arguments, and Decision. We are interested in the Decision part.

    APPROACH TO ANALYSIS:
    1. Read the paragraph carefully to understand the context.
    2. Divide the case law into sections: Facts, Issues, Decision. One section is comprise of consecutive paragraphs.
    3. Focus on the Decision section for identifying the application of law.
    4. In this section find the paragraphs, where the statutory legal provisions are applied to the specific facts by the Judge.
    5. Marked these paragraghs as if_law_applied True, marked all remaining as False.

    Make sure to read the whole case law before making a decision.
     return me the JSON object with the following fields:
    - para_id: Unique identifier for the paragraph in the case law text.
    - if_law_applied: Indicates whether this paragraph demonstrates an application of law to the facts.
    - section: The section of the case law where this paragraph is located (Facts, Issues, Decision).
   
    always return me a valid json list with all the paragraph_ids and their labels
    """
    # Create the example prompt template
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "para_id: {para_id}\npara_content: {para_content}"),
        
        ("ai", "para_id: {para_id}\nif_law_applied: {if_law_applied}\napplication_of_law_phrases: {application_of_law_phrases}")
    ])
    
    # Create the few-shot template
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt
    )
    
    
    # Create the final prompt template
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt2),
        few_shot_prompt,
        ("user", "{case_law}"),
        ])
    
    # Use ModelSelector to get the LLM
    llm = ModelSelector.get_llm('gpt-4.1-mini')
    
    # Create the chain
    chain = final_prompt | llm 
    response = chain.invoke({"case_law": case_law_json_list})

    return response


class Application(BaseModel):
    """
    Represents the classification outcome for a given paragraph in a case law text,
    indicating whether and how the law is applied to the specific facts.
    """
    
    para_id: str = Field(
        description="Unique identifier for the paragraph in the case law text."
    )
    if_law_applied: bool = Field(
        description="Indicates whether this paragraph demonstrates an application of law to the facts."
    )
    application_of_law_phrases: list[str] = Field(
        description="Collection of phrases or sentences showing how legal principles are applied in this paragraph."
    )
    reason: str = Field(
        description="Short explanation outlining why this paragraph does or does not demonstrate an application of law."
    )
def getLegalClassifierChain(examples, llm_type="gpt-4o-mini", temperature=0):
    """
    Creates and returns a classifier chain for analyzing legal text paragraphs.
    
    Args:
        examples: List of example classifications to use for few-shot learning
        llm_type: Model to use (gpt-4o-mini, gpt-4o, gpt-3.5-turbo, claude-3.5-sonnet, llama-3)
        temperature: Model temperature setting (default: 0)
    
    Returns:
        parser: The JsonOutputParser configured with the Application schema
        chain: The complete classification chain
    """
    # Initialize the appropriate LLM based on type
    if ModelSelector.MODEL_TYPE in ["deepseek-8b", "deepseek-v3-13b"]:
        # Use the DeepSeek model from ModelSelector
        llm = ModelSelector.get_llm(temperature=temperature)
    elif "deepseek" in llm_type:
        # Use the DeepSeek model from ModelSelector
        llm = ModelSelector.get_llm(temperature=temperature)
    elif "gpt" in llm_type:
        llm = ChatOpenAI(model=llm_type, temperature=temperature)
    elif "claude" in llm_type:
        llm = ChatAnthropic(model=llm_type, temperature=temperature)
    elif "llama" in llm_type:
        # Using specific Llama integration - adjust based on your implementation
        llm = ChatGroq(model=llm_type, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    system_prompt2 = """
    You are analyzing paragraphs from UK case law to determine if they contain an application of law to specific facts.

    APPLICATION OF LAW DEFINITION:
    An application of law is where statutory legal provisions are applied to the specific facts of the case at hand. This goes beyond merely citing or discussing law in the abstract and without specific reference to the facts of the case at hand.


    INDICATORS OF APPLICATION OF LAW:
    1. The judge connects specific statutory legal provisions to the specific factual circumstances.  
    2. The text shows reasoning that explains how the law resolves or addresses the unique facts
    3. The paragraph contains the judge's analysis leading to a conclusion based on legal principles
    4. Legal tests or criteria are being applied to the case facts

    NOT APPLICATIONS OF LAW:
    1. Mere citations of statutes, cases, or legal principles without application to facts
    2. Background procedural information or case history
    3. Statements about jurisdiction or general legal explanations
    4. Summaries of arguments made by parties without judicial analysis
    5. Restatements of previous cases without connecting them to current facts

    For each paragraph, determine if it contains an application of law, identify the specific phrases showing application, and provide a brief explanation for your decision.

    return a valid json, no prefix like this '''json before or end
    """
    # Combined and improved system prompt
    system_prompt = """
    You are analyzing paragraphs from UK case law to determine if they contain an application of law to specific facts.

    APPLICATION OF LAW DEFINITION:
    An application of law is where legal principles, statutes, or precedents are directly applied to the specific facts of the case at hand. This goes beyond merely citing or discussing law in the abstract.

    INDICATORS OF APPLICATION OF LAW:
    1. The judge explicitly connects legal rules/principles to the specific factual circumstances
    2. The text shows reasoning that explains how the law resolves or addresses the unique facts
    3. The paragraph contains the judge's analysis leading to a conclusion based on legal principles
    4. Legal tests or criteria are being applied to the case facts

    NOT APPLICATIONS OF LAW:
    1. Mere citations of statutes, cases, or legal principles without application to facts
    2. Background procedural information or case history
    3. Statements about jurisdiction or general legal explanations
    4. Summaries of arguments made by parties without judicial analysis
    5. Restatements of previous cases without connecting them to current facts

    For each paragraph, determine if it contains an application of law, identify the specific phrases showing application, and provide a brief explanation for your decision.
    """
    
    # Create the example prompt template
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "para_id: {para_id}\npara_content: {para_content}"),
        ("ai", "para_id: {para_id}\nif_law_applied: {if_law_applied}\napplication_of_law_phrases: {application_of_law_phrases}"
        #"\nreason: {reason}"
        )
    ])
    
    # Create the few-shot template
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt
    )
    
    # Create the final prompt template
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt2),
        few_shot_prompt,
        ("human", "para_id: {para_id}\npara_content: {para_content}\n{format_instructions}")
        ])
    
    # Create the parser
    parser = JsonOutputParser(pydantic_object=Application)
    
    # Create the chain
    chain = final_prompt | llm | parser
    
    return parser, chain

def getPhraseExtractionChain():
    system_prompt = """You are a specialized legal analyst with expertise in matching legal application between case law and legislation. Follow this systematic process:

    1. ANALYSIS PHASE:
      - Identify specific (not overly general) legal concepts or phrases in the case law.
      - Find the corresponding, equally specific portion in the legislation. This should be a somewhat longer, context-providing phrase.
      - From that longer legislative phrase, also extract the *key noun phrase(s)* or *core concept(s)*—the minimal expression that captures the critical legal idea.

    2. MATCHING CRITERIA:
      - Direct textual overlap or near-verbatim references (no paraphrasing).
      - Semantic equivalence in the same legal context (avoid purely generic wording).
      - Clear interpretative relationship (case law explains or applies the legislation).
      - Substantive connection (not merely tangential mentions).

    3. VALIDATION RULES:
      - **Only** extract text that actually appears in each source (verbatim).
      - For `"legislation_term"`, an excerpt from legislation text use the longer snippet that captures context.
      - For `"key_phrases/concepts"`, extract the essential, shorter noun phrase(s) from within that legislation snippet.
      - Ensure the match has legal interpretive or explanatory value (avoid trivial or broad phrases).

    4. OUTPUT STRUCTURE:
      Return a **JSON array** of objects. Each object must contain:
      - `"caselaw_term"`: exact phrase/excerpt from the case law (no rewording).
      - `"legislation_term"`: a longer, context-inclusive phrase/excerpt from the legislation.
      - `"key_phrases/concepts"`: the shorter core phrase(s)—verbatim—taken from within `"legislation_term"` that most directly capture the legal concept (often a noun phrase).
      - `"reasoning"`: brief explanation of how the case law term interprets/applies the legislation.
      - `"confidence"`: "High", "Medium", or "Low" based on how closely they match in legal meaning.

    Example Output:
    [
      {{
        "caselaw_term": "reasonable safety standards required documented weekly inspections",
        "para_id": "ewhc_11",
        "section_id": "20",
        "legislation_term": "reasonable safety standards",
        "key_phrases":["reasoable safety standards"],
        "reasoning": "Case law directly interprets and defines the legislative phrase",
        "confidence": "High"
      }}
    ]

    Example Input:
    Legislation : "If a parent does not provide proper care and guardianship for a child, the local authority may intervene to ensure the child's welfare is safeguarded. A person is eligible for legal aid if they cannot afford legal representation and the matter pertains to family law or children's welfare."
    Case Law: "The parent failed to ensure the child received proper care, necessitating intervention by the local authority. Legal aid is required in this case due to the individual's inability to afford representation in a family law matter."
    "para_id": "ewhc_11",
    "section_id": "20",



    Example Output:
    [
      {{
        "para_id": "ewhc_11",
        "section_id": "20",
        "case_law_term": "parent failed to ensure the child received proper care",
        "legislation_term": "parent does not provide proper care and guardianship for a child",
        "key_phrases":["proper care and guardianship for a child"],
        "reasoning": "Direct interpretation of parental care obligation",
        "confidence": "High"
      }},
      {{
        "para_id": "ewhc_11",
        "section_id": "20",
        "case_law_term": "individual's inability to afford representation in a family law matter",
        "legislation_term": "person is eligible for legal aid if they cannot afford legal representation and the matter pertains to family law",
        "key_phrases":["legal representation"],
        "reasoning": "Case applies legislative criteria for legal aid eligibility",
        "confidence": "High"
      }}
    ]

    Rules:
    - Extract only exact phrases from source texts
    - No rephrasing or inference
    - Include only paired matches with clear legal interpretation
    - Return raw JSON without formatting or explanation
    - ALWAYS RETURN SOME RESULT !!!

    """

    human_prompt = """Process these legal texts following the above methodology:

    Legislation:
    {legislation_text}

    Case Law:
    {case_text}

    Paragraph ID:
    {para_id}

    Legislation Section ID:
    {section_id}

    There should be nothing produced from your own side -- Just extract from the given sections of caselaw and legislation.
    Don't return back the input. 
    Return only the JSON array with matches.
    Include reasoning and confidence scores.
    ALWAYS RETURN SOME RESULT !!!
    """

    prompt_template3 = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt)
    ])
    
    # Use ModelSelector to get the LLM
    # If using a fine-tuned model, we'll need to handle it differently
    if ModelSelector.MODEL_TYPE in ["deepseek-v3-8b", "deepseek-v3-13b"]:
        llm = ModelSelector.get_llm(temperature=0)
    else:
        # For OpenAI, we can use the fine-tuned model
        model = "ft:gpt-4o-mini-2024-07-18:swansea-university::B3pbF9HD"  # "gpt-4o"
        llm = ChatOpenAI(model=model, temperature=0)
    
    llm_chain_extraction = LLMChain(llm=llm, prompt=prompt_template3)
    return llm_chain_extraction
    
def getEmbeddings():
    print(OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings()
    return embeddings

def getInterPretations(legislation_text, case_text,para_id, section_id, llm_chain_extraction):
    input_data = {
        "legislation_text": legislation_text,
        "case_text": case_text,
        "para_id": para_id,
        "section_id": section_id
    }
    
    # Run the LLM chain
    result = llm_chain_extraction.invoke(input_data)
    try:
        return result['text']
    except:
        return result
def BuildVectorDB(directory, legislation_list):
    """
    Build a vector database from legislation documents with batched processing
    to handle token limit constraints.
    """
    def load_legislative_sections(directory, legislation_number):
        sections = []
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(directory, filename), 'r') as file:
                        text = file.read().strip()  # Read the content of the file
                        sections.append({
                            "id": f"{legislation_number}_{filename.split('.')[0]}",
                            "text": text,
                            "legislation_id": legislation_number
                        })
                except:
                    pass
        return sections

    docs = []
    for legislation_number in legislation_list:
        try:
            legislative_sections = load_legislative_sections(
                f"{directory}/{legislation_number}", legislation_number
            )
            doc = [
                Document(page_content=sec["text"], 
                         metadata={
                             "id": sec["id"],
                             "legislation_id": sec["legislation_id"]
                         }) 
                for sec in legislative_sections
            ]
            docs.extend(doc)
        except Exception as e:
            print(f"Error processing legislation {legislation_number}: {str(e)}")
    
    # Now use batched processing to create embeddings
    try:
        vectorstore = build_vector_db_with_batching(docs)
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        raise

def build_vector_db_with_batching(docs: list[Document], 
                                  batch_size: int = 100, 
                                  max_retries: int = 5,
                                  retry_sleep: int = 2) -> FAISS:
    """
    Build a FAISS vector store from documents using batched embedding to avoid token limits.
    
    Args:
        docs: List of documents to embed
        batch_size: Initial batch size (number of documents)
        max_retries: Maximum number of retries for each batch
        retry_sleep: Seconds to sleep between retries
    
    Returns:
        FAISS vector store with embedded documents
    """
    embeddings = getEmbeddings()
    
    all_embeddings = []
    all_texts = []
    all_metadatas = []
    
    i = 0
    current_batch_size = batch_size
    
    while i < len(docs):
        end_idx = min(i + current_batch_size, len(docs))
        batch = docs[i:end_idx]
        
        batch_texts = [doc.page_content for doc in batch]
        batch_metadatas = [doc.metadata for doc in batch]
        
        retry_count = 0
        success = False
        
        while not success and retry_count < max_retries:
            try:
                print(f"Processing batch {i} to {end_idx} ({len(batch_texts)} documents)")
                batch_embeddings = embeddings.embed_documents(batch_texts)
                
                all_embeddings.extend(batch_embeddings)
                all_texts.extend(batch_texts)
                all_metadatas.extend(batch_metadatas)
                
                print(f"Successfully embedded batch {i} to {end_idx} of {len(docs)}")
                success = True
                
            except Exception as e:
                error_msg = str(e)
                retry_count += 1
                
                if "max_tokens_per_request" in error_msg:
                    current_batch_size = max(1, current_batch_size // 2)
                    end_idx = min(i + current_batch_size, len(docs))
                    batch = docs[i:end_idx]
                    batch_texts = [doc.page_content for doc in batch]
                    batch_metadatas = [doc.metadata for doc in batch]
                    
                    print(f"Token limit exceeded. Reducing batch size to {current_batch_size} and retrying...")
                else:
                    print(f"Error in batch {i} to {end_idx}: {error_msg}")
                    print(f"Retry {retry_count}/{max_retries}...")
                
                time.sleep(retry_sleep)
        
        if success:
            i = end_idx
        else:
            print(f"Failed to process batch after {max_retries} retries. Skipping to next batch.")
            i = end_idx
    
    if not all_embeddings:
        raise ValueError("No documents were successfully embedded")
    
    print(f"Creating FAISS index with {len(all_embeddings)} embeddings")
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(all_texts, all_embeddings)),
        embedding=embeddings,
        metadatas=all_metadatas
    )
    
    return vector_store


if __name__ == "__main__":
    print("OpenAI Handler")
    ModelSelector.MODEL_TYPE = "deepseek-qwen-32b"

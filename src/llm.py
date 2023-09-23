'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
from langchain.llms import CTransformers
from dotenv import find_dotenv, load_dotenv
import box
import yaml
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def build_llm():
    # Local CTransformers model
#     llm = CTransformers(model=cfg.MODEL_BIN_PATH,
#                         model_type=cfg.MODEL_TYPE,
#                         config={'max_new_tokens': cfg.MAX_NEW_TOKENS,
#                                 'temperature': cfg.TEMPERATURE}
#                         )
    model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_BIN_PATH,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    model = model.eval()

    return model

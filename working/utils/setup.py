from omegaconf import OmegaConf
from pathlib import Path
import os

def setup():
    config = OmegaConf.load(Path('./working/config.yml'))
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = config.deepseek_api
    
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = config.gemini_api

    if not os.environ.get("QWEN_API_KEY"):
        os.environ["QWEN_API_KEY"] = config.deepinfra_api

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = config.openai_api
 
    if not os.environ.get("DEEPINFRA_API_KEY"):
        os.environ["DEEPINFRA_API_KEY"] = config.deepinfra_api

    return config






def main():
    print("Setup script executed. Add any setup tasks here.")

if __name__ == "__main__":
    main()
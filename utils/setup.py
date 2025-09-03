from omegaconf import OmegaConf
import os

def setup():
    config = OmegaConf.load('config.yml')
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = config.deepseek_api






def main():
    print("Setup script executed. Add any setup tasks here.")

if __name__ == "__main__":
    main()
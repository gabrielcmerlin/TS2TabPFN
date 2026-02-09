import argparse
import yaml

import argparse
import yaml

class Parser:
    """
    Handles command-line argument parsing and loads experiment configuration 
    from a YAML file.
    
    This class uses argparse to parse the '--config' argument, which should 
    point to a YAML configuration file. The YAML file is then loaded and 
    returned as a Python dictionary.
    """

    def __init__(self):
        """
        Initializes the argument parser with a required '--config' argument.
        """

        self.parser = argparse.ArgumentParser(description="Load experiment configuration")
        self.parser.add_argument(
            "-c", "--config", required=True, help="Path to config.yaml"
        )

    def parse(self):
        """
        Parses command-line arguments and loads the configuration from the 
        specified YAML file.
        """
        
        args = self.parser.parse_args()
        
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        return config
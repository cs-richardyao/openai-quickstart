import argparse
import os


class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Translate English PDF book to Chinese.')
        self.parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file with model and API settings.')
        self.parser.add_argument('--model_type', default='OpenAIModel', type=str, choices=['GLMModel', 'OpenAIModel'], help='The type of translation model to use. Choose between "GLMModel" and "OpenAIModel".')
        self.parser.add_argument('--glm_model_url', type=str, help='The URL of the ChatGLM model URL.')
        self.parser.add_argument('--timeout', type=int, help='Timeout for the API request in seconds.')
        self.parser.add_argument('--openai_model', type=str, help='The model name of OpenAI Model. Required if model_type is "OpenAIModel".')
        self.parser.add_argument('--openai_api_key', default=os.getenv('OPENAI_KEY'), type=str, help='The API key for OpenAIModel. Required if model_type is "OpenAIModel".')
        self.parser.add_argument('--openai_base_url', default=os.getenv('OPENAI_BASE'),  type=str, help='The API key for OpenAIModel. Required if model_type is "OpenAIModel".')
        self.parser.add_argument('--book', type=str, help='PDF file to translate.')
        self.parser.add_argument('--file_format', type=str, help='The file format of translated book. Now supporting PDF and Markdown')
        self.parser.add_argument('--origin_language', default='中文', type=str, help='The original language to translate to. Default is Chinese.')
        self.parser.add_argument('--target_language', default='中文', type=str, help='The target language to translate to. Default is Chinese.')
        self.parser.add_argument('--output_file', default='test', type=str, help='The target language to translate to. Default is Chinese.')

    def parse_arguments(self):
        args = self.parser.parse_args()
        if args.model_type == 'OpenAIModel' and not args.openai_model and not args.openai_api_key:
            self.parser.error("--openai_model and --openai_api_key is required when using OpenAIModel")
        return args

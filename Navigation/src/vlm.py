import logging
import os
import torch
import numpy as np
import cv2
import base64
import requests
import io
from PIL import Image
from typing import Optional, List
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, pipeline
import time
from openai import AzureOpenAI, OpenAI

class VLM:
    """
    Base class for a Vision-Language Model (VLM) agent. 
    This class should be extended to implement specific VLMs.
    """

    def __init__(self, **kwargs):
        """
        Initializes the VLM agent with optional parameters.
        """
        self.name = "not implemented"

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform inference with the VLM agent, passing images and a text prompt.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError
    
    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the VLM, incorporating past context.

        Parameters
        ----------
        history : int
            The number of context steps to keep for inference.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def rewind(self):
        """
        Rewind the VLM agent one step by removing the last inference context.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total cost or spend associated with the agent.
        """
        return 0


class OtherVLM(VLM):
    """
    A specific implementation of a VLM using the Gemini API for image and text inference.
    """
    def __init__(self, model="grok-3",  base_url="http://localhost:8000/v1/chat/completions", system_instruction=None):
        self.name = model
        self.headers = {
            'Authorization': 'Bearer ',
            'Content-Type': 'application/json',
        }
        endpoint = ""
        self.model_name = "o4-mini"
        self.deployment = "o4-mini"
        print(f"OtherVLM init : model name-{self.model_name}, url-{endpoint}")
        subscription_key = ""
        api_version = ""
        self.client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )

        self.spend = 0
        if system_instruction:
            self.system_instruction = system_instruction


    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        max_retries = 10000 
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                
                content = [{"type": "text", "text": text_prompt}]
                content.extend(self._convert_images_to_base64(images))
                
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=[
                        {"role": "system", "content": self.system_instruction},
                        {"role": "user", "content": content}
                    ]
                )
                response_text = response.choices[0].message.content
                
                return response_text
                    
            except Exception as e:
                logging.error(f"API ERROR (Attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count == max_retries:
                    return "API ERROR"
                time.sleep(1)

    def _convert_images_to_base64(self, images):
        content = []
        for i, image in enumerate(images):
            try:
                if image is None:
                    logging.warning(f"Skipping None image at index {i}")
                    continue

                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=2)
                elif len(image.shape) == 3 and image.shape[2] != 3:
                    image = image[:, :, :3]

                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                    
                # logging.info(f"Final image {i} shape: {image.shape}")

                pil_image = Image.fromarray(image)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_str}"
                    }
                })
                
            except Exception as e:
                logging.error(f"Error converting image {i}: {str(e)}")
                logging.error(f"Image info: shape={image.shape if image is not None else 'None'}, "
                            f"dtype={image.dtype if image is not None else 'None'}")
                
        return content

    def call(self, images: list[np.array], text_prompt: str, type: str=None):
        return self.call_chat(0, images, text_prompt)

    def reset(self):
        self.conversation_history = []

    def rewind(self):
        if len(self.conversation_history) >= 2:
            self.conversation_history = self.conversation_history[:-2]

    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend



class QwenVLM(VLM):

    def __init__(self, model="", base_url="", system_instruction=None):

        self.name = model
        self.model_name = model
        self.url = base_url
        print(f"QwenVLM init : model name-{self.model_name}, url-{self.url}")
        self.system_instruction = system_instruction

        self.conversation_history = []
        if system_instruction:
            self.conversation_history.append({
                "role": "system",
                "content": system_instruction
            })
        self.spend = 0  
        
    def _encode_image(self, image):
        pil_image = Image.fromarray(image[:, :, :3])
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        img_bytes = buffer.getvalue()
        
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def call(self, images: List[np.array]=None, text_prompt: str=None, type: str=None):
        max_retries = 1000
        retry_count = 0

        user_content = []
        if type == 'image':
            for image in images:
                base64_image = self._encode_image(image)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        elif type == 'ge_obs': 
            self.system_instruction = '''
                You are an indoor navigation assistant. Your task is to help me understand my current location and the direction I am moving in. In each turn, you will receive two images:
                1. **Observation Image (obs)**: This shows a first-person view of my surroundings.
                2. **Action-Selection Overlay (arrow image)**: This indicates the direction I am choosing to move.
                Your job is to:
                - Identify the room I am currently in based on the observation image.
                - Pick out up to three iconic objects in the room (prioritize furniture and ignore the floor).
                - Determine where the green arrow in the overlay is pointing.
                Provide your response in a clear and concise manner.
            '''
            # Process all images in the list
            for image in images:
                base64_image = self._encode_image(image)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        elif type == 'ge_history':
            self.system_instruction = (
                "You are an assistant skilled at summarizing the user's input. "
                "Your task is to take a sequence of descriptions about the user's movements and observations, and condense them into a concise and coherent summary of their movement history."
                "Focus on capturing the key locations visited and notable observations made during each step."
            )

        user_content.append({
            "type": "text",
            "text": text_prompt
        })

        payload = {
            "model": self.model_name,
            "messages": [
            {
            "role": "system",
            "content": f"{self.system_instruction}"
            },
            {
            "role": "user",
            "content": user_content
            }
        ]
        }
        
        headers = {
        "Content-Type": "application/json"
        }

        while retry_count < max_retries:
            try:
                response = requests.request("POST", self.url, json=payload, headers=headers,proxies={"http": None, "https": None, "all": None})
                if response.status_code == 200:
                    response_data = response.json()
                    return response_data["choices"][0]["message"]["content"]
                logging.error(f"API ERROR (Attempt {retry_count + 1}/{max_retries}): {response.text}")
                retry_count += 1
                if retry_count == max_retries:
                    return "QWEN API ERROR"
                time.sleep(0.5)  
            except Exception as e:
                logging.error(f"QWEN API ERROR (Attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count == max_retries:
                    return f"QWEN API EROOR: {str(e)}"
                time.sleep(1)
    
    def call_chat(self, images: List[np.array]=None, text_prompt: str=None, type: str=None):
        return self.call_chat(images, text_prompt, "image")
    
    def reset(self):
        if self.system_instruction:
            self.conversation_history = [{
                "role": "system",
                "content": self.system_instruction
            }]
        else:
            self.conversation_history = []
    
    def rewind(self):
        if len(self.conversation_history) >= 2:
            self.conversation_history = self.conversation_history[:-2]
    
    def get_spend(self):
        return self.spend
  
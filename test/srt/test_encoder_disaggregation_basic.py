import json
import os
import unittest
from types import SimpleNamespace

import openai
import requests
from transformers import AutoTokenizer

from sglang.test.test_encoder_disaggregation_utils import TestEncoderDisaggregationBase
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    DEFAULT_URL_FOR_TEST,
)


class TestDisaggregationAccuracy(TestEncoderDisaggregationBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = "/disk3/models/Qwen2.5-VL-7B-Instruct/"
        cls.base_url = DEFAULT_URL_FOR_TEST

        # Non blocking start servers
        cls.start_prefill()
        cls.start_decode()
        cls.start_encoder()

        # Block until both
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")
        cls.wait_server_ready(cls.encoder_url + "/health")

        cls.launch_lb()

    
    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp",
            "1",
            "--language-only",
            f"--encoder-urls",
            f"{cls.encoder_url}",
        ]
        prefill_args += ["--disaggregation-transfer-backend" ,"nixl"]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
        ]
        decode_args += ["--disaggregation-transfer-backend" ,"nixl"]
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )
    
    @classmethod
    def start_encoder(cls):
        encoder_args = [
            "--encoder-only",
            "--base-gpu-id",
            "2",
        ]
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.encoder_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encoder_args,
        )

    def verify_single_image_response(self, response):
        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)

        # `driver` is for gemma-3-it
        assert (
            "man" in text or "person" or "driver" in text
        ), f"text: {text}, should contain man, person or driver"
        assert (
            "cab" in text
            or "taxi" in text
            or "SUV" in text
            or "vehicle" in text
            or "car" in text
        ), f"text: {text}, should contain cab, taxi, SUV, vehicle or car"
        # MiniCPMO fails to recognize `iron`, but `hanging`
        assert (
            "iron" in text or "hang" in text or "cloth" in text or "holding" in text
        ), f"text: {text}, should contain iron, hang, cloth or holding"
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_single_image_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "/disk3/lsy/sglang-colocate/.vscode/man_ironing_on_back_of_suv.png"},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
        )

        print("-" * 30)
        print(f"Single image response:\n{response.choices[0].message.content}")
        print("-" * 30)

        self.verify_single_image_response(response)
        
        



if __name__ == "__main__":
    unittest.main()

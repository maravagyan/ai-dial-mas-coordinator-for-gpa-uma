# task/agent.py

import json
import os
from typing import Any, Optional

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, Stage

from task.coordination.gpa import GPAGateway
from task.coordination.ums_agent import UMSAgentGateway
from task.logging_config import get_logger
from task.models import CoordinationRequest, AgentName
from task.prompts import COORDINATION_REQUEST_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
from task.stage_util import StageProcessor

logger = get_logger(__name__)


class MASCoordinator:
    def __init__(
        self,
        endpoint: str,
        deployment_name: str,
        ums_agent_endpoint: str,
        gpa_endpoint: str,
    ):
        self.endpoint = endpoint  # DIAL Core URL (router + final)
        self.deployment_name = deployment_name  # usually "gpt-4o"
        self.ums_agent_endpoint = ums_agent_endpoint
        self.gpa_endpoint = gpa_endpoint  # GPA service URL (http://localhost:8052)

    async def handle_request(self, choice: Choice, request: Request) -> Message:
        # 1) Create AsyncDial client for DIAL Core
        dial_api_key = os.getenv("DIAL_API_KEY")
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=dial_api_key,
            api_version="2025-01-01-preview",
        )

        # 2) Open stage for Coordination Request
        coord_stage: Stage = StageProcessor.open_stage(choice, "Coordination Request")

        # 3) Prepare coordination request (LLM router)
        coordination_request = await self.__prepare_coordination_request(client, request)

        # 4) Add generated coordination request to the stage and close it
        coord_stage.append_content(coordination_request.model_dump_json(indent=2))
        StageProcessor.close_stage_safely(coord_stage)

        # IMPORTANT: field name is agent_name (not agent)
        agent_name = coordination_request.agent_name

        # 5) Handle coordination request (agent execution stage)
        exec_stage: Stage = StageProcessor.open_stage(
            choice, f"Calling {agent_name.value.upper()} Agent"
        )

        agent_message = await self.__handle_coordination_request(
            coordination_request=coordination_request,
            choice=choice,
            stage=exec_stage,
            request=request,
        )
        StageProcessor.close_stage_safely(exec_stage)

        # 6) Generate final response based on the message from called agent
        final_stage: Stage = StageProcessor.open_stage(choice, "Final Response")
        final_message = await self.__final_response(
            client=client,
            choice=choice,
            request=request,
            agent_message=agent_message,
        )
        StageProcessor.close_stage_safely(final_stage)

        return final_message

    async def __prepare_coordination_request(
        self, client: AsyncDial, request: Request
    ) -> CoordinationRequest:
        messages = self.__prepare_messages(request, COORDINATION_REQUEST_SYSTEM_PROMPT)

        # Structured outputs
        extra_body = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": CoordinationRequest.model_json_schema(),
                },
            }
        }

        resp = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=messages,
            stream=False,
            extra_body=extra_body,
        )

        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        return CoordinationRequest.model_validate(data)

    def __prepare_messages(self, request: Request, system_prompt: str) -> list[dict[str, Any]]:
        res: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

        for msg in request.messages:
            # If user message has custom_content, keep only plain text
            if msg.role == Role.USER and msg.custom_content is not None:
                res.append({"role": "user", "content": msg.content or ""})
                continue

            # Otherwise append as dict excluding None
            res.append(msg.dict(exclude_none=True))

        return res

    async def __handle_coordination_request(
        self,
        coordination_request: CoordinationRequest,
        choice: Choice,
        stage: Stage,
        request: Request,
    ) -> Message:
        additional_instructions: Optional[str] = coordination_request.additional_instructions

        # IMPORTANT: field name is agent_name
        if coordination_request.agent_name == AgentName.UMS:
            ums = UMSAgentGateway(self.ums_agent_endpoint)
            return await ums.response(
                choice=choice,
                stage=stage,
                request=request,
                additional_instructions=additional_instructions,
            )

        # default: GPA
        gpa = GPAGateway(endpoint=self.gpa_endpoint)
        return await gpa.response(
            choice=choice,
            stage=stage,
            request=request,
            additional_instructions=additional_instructions,
        )

    async def __final_response(
        self,
        client: AsyncDial,
        choice: Choice,
        request: Request,
        agent_message: Message,
    ) -> Message:
        messages = self.__prepare_messages(request, FINAL_RESPONSE_SYSTEM_PROMPT)

        user_request = ""
        if request.messages and request.messages[-1].role == Role.USER:
            user_request = request.messages[-1].content or ""

        augmented = (
            "CONTEXT:\n"
            f"{agent_message.content or ''}\n\n"
            "USER_REQUEST:\n"
            f"{user_request}"
        )

        # Replace last user message content with augmented prompt
        if messages:
            messages[-1]["content"] = augmented

        content_acc = ""

        stream = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                content_acc += delta.content
                choice.append_content(delta.content)

        return Message(role=Role.ASSISTANT, content=content_acc)
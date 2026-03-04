from copy import deepcopy
from typing import Optional, Any
import os

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Role, Choice, Request, Message, CustomContent, Stage, Attachment

from task.stage_util import StageProcessor

_IS_GPA = "is_gpa"
_GPA_MESSAGES = "gpa_messages"

GPA_DEPLOYMENT_NAME = os.getenv("GPA_DEPLOYMENT_NAME", "gpa")


class GPAGateway:

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    async def response(
            self,
            choice: Choice,
            stage: Stage,
            request: Request,
            additional_instructions: Optional[str]
    ) -> Message:
        dial_api_key = os.getenv("DIAL_API_KEY")
        client = AsyncDial(
            endpoint=self.endpoint,
            api_key=dial_api_key,
            api_version="2025-01-01-preview",
        )

        messages = self.__prepare_gpa_messages(request, additional_instructions)

        # required for GPA RAG logic
        extra_headers = {"x-conversation-id": request.headers.get("x-conversation-id")}

        stream = await client.chat.completions.create(
            deployment_name=GPA_DEPLOYMENT_NAME,
            messages=messages,
            stream=True,
            extra_headers=extra_headers,
        )

        content = ""
        result_custom_content = CustomContent(attachments=[], state={})
        stages_map: dict[int, Stage] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta:
                continue

            # helpful debug
            if delta.content:
                print(delta.content, end="", flush=True)

            # content → append to this stage
            if delta.content:
                content += delta.content
                stage.append_content(delta.content)

            # custom_content magic: attachments/state/stages propagation
            if delta.custom_content:
                cc = delta.custom_content

                if cc.attachments:
                    result_custom_content.attachments.extend(cc.attachments)

                if cc.state:
                    # merge state
                    if result_custom_content.state is None:
                        result_custom_content.state = {}
                    result_custom_content.state.update(cc.state)

                cc_dict = cc.dict(exclude_none=True)
                if "stages" in cc_dict:
                    for stg in cc_dict["stages"]:
                        idx = stg.get("index")
                        if idx is None:
                            continue

                        if idx in stages_map:
                            local_stage = stages_map[idx]
                            if stg.get("content"):
                                local_stage.append_content(stg["content"])
                            if stg.get("attachments"):
                                for att in stg["attachments"]:
                                    local_stage.add_attachment(Attachment(**att))
                            if stg.get("status") == "completed":
                                StageProcessor.close_stage_safely(local_stage)
                        else:
                            local_stage = StageProcessor.open_stage(choice, stg.get("name", f"GPA Stage {idx}"))
                            stages_map[idx] = local_stage
                            if stg.get("content"):
                                local_stage.append_content(stg["content"])
                            if stg.get("attachments"):
                                for att in stg["attachments"]:
                                    local_stage.add_attachment(Attachment(**att))
                            if stg.get("status") == "completed":
                                StageProcessor.close_stage_safely(local_stage)

        # 5) Propagate collected custom_content to choice
        # Convert attachments if needed (SDK vs client model mismatch)
        if result_custom_content.attachments:
            converted = []
            for a in result_custom_content.attachments:
                converted.append(Attachment(**a.dict(exclude_none=True)))
            result_custom_content.attachments = converted

        choice.set_custom_content(result_custom_content)

        # 6) Persist GPA state into choice state so we can restore tool context later
        choice.set_state({_IS_GPA: True, _GPA_MESSAGES: result_custom_content.state})

        # 7) return assistant message
        return Message(role=Role.ASSISTANT, content=content)

    def __prepare_gpa_messages(self, request: Request, additional_instructions: Optional[str]) -> list[dict[str, Any]]:
        res_messages: list[dict[str, Any]] = []

        # restore only GPA-related tool/state history
        for idx in range(len(request.messages)):
            msg = request.messages[idx]
            if msg.role == Role.ASSISTANT and msg.custom_content and msg.custom_content.state:
                st = msg.custom_content.state
                if st.get(_IS_GPA) is True:
                    # 1) add previous user message
                    if idx - 1 >= 0:
                        res_messages.append(request.messages[idx - 1].dict(exclude_none=True))

                    # 2) restore assistant message format for GPA
                    restored = deepcopy(msg)
                    restored.custom_content.state = st.get(_GPA_MESSAGES) or {}
                    res_messages.append(restored.dict(exclude_none=True))

        # add latest user message
        last_user = request.messages[-1]
        res_messages.append(last_user.dict(exclude_none=True))

        # augment last user message with additional instructions
        if additional_instructions:
            res_messages[-1]["content"] = (res_messages[-1].get("content") or "") + "\n\n" + additional_instructions

        return res_messages
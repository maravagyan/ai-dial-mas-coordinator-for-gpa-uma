import httpx
from aidial_sdk.chat_completion import Choice, Request, Message, Role, Stage

from task.stage_util import StageProcessor


class UMSAgentGateway:
    def __init__(self, ums_agent_endpoint: str):
        self.ums_agent_endpoint = ums_agent_endpoint.rstrip("/")

    async def response(
        self,
        choice: Choice,
        stage: Stage,
        request: Request,
        additional_instructions: str | None = None,
    ) -> Message:
        # take last user message
        user_text = ""
        if request.messages and request.messages[-1].role == Role.USER:
            user_text = request.messages[-1].content or ""

        if additional_instructions:
            user_text = f"{user_text}\n\nAdditional instructions:\n{additional_instructions}"

        conversation_id = await self.__create_ums_conversation()

        # Call /conversations/{id}/chat
        url = f"{self.ums_agent_endpoint}/conversations/{conversation_id}/chat"
        payload = {
            "message": {"role": "user", "content": user_text},
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            # Streaming response (SSE-like chunks)
            async with client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()

                content_acc = ""
                async for line in resp.aiter_lines():
                    # Some servers send empty keepalive lines
                    if not line:
                        continue

                    # If it is "data: ...." style, strip prefix
                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()

                    # Many implementations stream plain text chunks.
                    # If yours streams JSON, we can adjust after you paste 1 streamed line.
                    content_acc += line
                    choice.append_content(line)

        return Message(role=Role.ASSISTANT, content=content_acc)

    async def __create_ums_conversation(self) -> str:
        url = f"{self.ums_agent_endpoint}/conversations"
        payload = {}  # title is optional per your OpenAPI

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["id"]